#!/usr/bin/env python
import math, sys
import rospy
import actionlib
import PyKDL as kdl
import kdl_parser_py.urdf as kdl_parser
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from std_msgs.msg import Float64MultiArray, Float64
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, SwitchControllerResponse
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
import image_geometry
from sensor_msgs.msg import CameraInfo
import tf  # used to get position of the end effector
from scipy.signal import butter, lfilter, freqz
import numpy as np
import matplotlib.pyplot as plt
import cv2  # used to downsample and convert depth image to array
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import torch
from network.model import ObservationModelUNet, Motion2DModel


class URGripah(object):
    def __init__(self):

        self.base_link = 'base_link'
        self.ee_link = 'ee_link'
        flag, self.tree = kdl_parser.treeFromParam('/robot_description')
        self.chain = self.tree.getChain(self.base_link, self.ee_link)
        self.num_joints = self.tree.getNrOfJoints()
        self.vel_ik_solver = kdl.ChainIkSolverVel_pinv(self.chain)
        self.pos_fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.pos_ik_solver = kdl.ChainIkSolverPos_LMA(self.chain)
        
        self.cam_model = image_geometry.PinholeCameraModel()
        
        self.joint_state = kdl.JntArrayVel(self.num_joints)
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        rospy.init_node('ur_eef_vel_controller')
        rospy.Subscriber('/joint_states', JointState, self.arm_joint_state_cb)
        rospy.Subscriber('/rdda_interface/joint_states', JointState, self.finger_joint_state_cb)
        self.joint_vel_cmd_pub = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray, queue_size=10)
        self.joint_pos_cmd_pub = rospy.Publisher('/scaled_pos_traj_controller/command', JointTrajectory, queue_size=10)
        self.speed_scaling_pub = rospy.Publisher('/speed_scaling_factor', Float64, queue_size=10)

        self.switch_controller_cli = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        self.joint_pos_cli = actionlib.SimpleActionClient('/scaled_pos_joint_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        cam_info = rospy.wait_for_message('/camera/depth/camera_info', CameraInfo, timeout=10)
        self.cam_model.fromCameraInfo(cam_info)

        self.bridge = CvBridge() # allows conversion from depth image to array
        self.tf_listener = tf.TransformListener()
        self.image_offset = 100
        self.pos_ctrler_running = False
        rospy.sleep(0.5)

    def switch_controller(self, mode=None):
        req = SwitchControllerRequest()
        res = SwitchControllerResponse()

        req.start_asap = False
        req.timeout = 0.0
        if mode == 'velocity' and self.pos_ctrler_running:
            req.start_controllers = ['joint_group_vel_controller']
            req.stop_controllers = ['scaled_pos_joint_traj_controller']
            req.strictness = req.STRICT
            self.pos_ctrler_running = False
        elif mode == 'position' and not self.pos_ctrler_running:
            req.start_controllers = ['scaled_pos_joint_traj_controller']
            req.stop_controllers = ['joint_group_vel_controller']
            req.strictness = req.STRICT
            self.pos_ctrler_running = True
        # else:
        #     rospy.logwarn('Unkown mode for the controller!')

        res = self.switch_controller_cli.call(req)

    def step(self, action):
        eef_vel = [0.0, -0.02, 0.0]
        sample_freq = 16
        num_steps = 200
        rate = rospy.Rate(64)
        finger_pos = np.zeros((num_steps, 2))
        states = np.zeros((num_steps // sample_freq, 2), dtype=np.long)
        finger_start_pos = self.joint_pos

        sample_n = 0
        for t in range(num_steps):
            if rospy.is_shutdown():
                break
            
            q_dot = self._calculate_qdot(eef_vel)
            self.pub_joint_vel(q_dot)
            rate.sleep()
            finger_pos[t] = self.joint_pos - finger_start_pos
            if (t+1) % sample_freq == 0:
                states[sample_n] = self.get_current_pixel_coords()
                sample_n += 1
        self.pub_joint_vel([0.0] * 6)
        return finger_pos, states

    def run_line_traj(self, image=None):
        
        eef_vel = [0.0, -0.02, 0.0]
        sample_freq = 16
        num_steps = 2240
        # rate = rospy.Rate(64)
        rate = rospy.Rate(50)
        finger_pos = np.zeros((num_steps, 2))
        states = np.zeros((num_steps // sample_freq, 2), dtype=np.long)

        finger_start_pos = self.joint_pos

        n_freq = 0
        for t in range(num_steps):
            if rospy.is_shutdown():
                break

            q_dot = self._calculate_qdot(eef_vel)
            self.pub_joint_vel(q_dot)
            rate.sleep()
            finger_pos[t] = self.joint_pos - finger_start_pos
            if (t + 1) % sample_freq == 0:
                
                states[n_freq] = self.get_current_pixel_coords()
                print(states[n_freq])
                if image is not None:
                    image = np.array(image * 255, dtype=np.uint8)
                    image_vis = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
                    if np.all(states[n_freq] >= 0) and np.all(states[n_freq] < 64):
                        image_vis[states[n_freq][0], states[n_freq][1]] = [0, 0, 255]
                    image_vis = cv2.resize(image_vis, (64 * 5, 64 * 5), interpolation=cv2.INTER_NEAREST)

                    cv2.imshow('Depth Image', image_vis)
                    cv2.waitKey(1)
                n_freq += 1

        self.pub_joint_vel([0.0] * 6)
        return finger_pos, states

    def arm_joint_state_cb(self, joint_msg):
        self.joint_state.q[0] = joint_msg.position[2]
        self.joint_state.q[1] = joint_msg.position[1]
        self.joint_state.q[2] = joint_msg.position[0]
        self.joint_state.q[3] = joint_msg.position[3]
        self.joint_state.q[4] = joint_msg.position[4]
        self.joint_state.q[5] = joint_msg.position[5]

        self.joint_state.qdot[0] = joint_msg.velocity[2]
        self.joint_state.qdot[1] = joint_msg.velocity[1]
        self.joint_state.qdot[2] = joint_msg.velocity[0]
        self.joint_state.qdot[3] = joint_msg.velocity[3]
        self.joint_state.qdot[4] = joint_msg.velocity[4]
        self.joint_state.qdot[5] = joint_msg.velocity[5]

    def finger_joint_state_cb(self, joint_msg):
        self.finger_joint_state = joint_msg
        self.joint_pos = -np.array(joint_msg.position)

    def pub_joint_vel(self, q_dot):
        joint_vel = Float64MultiArray()
        qdot = [0.0]*self.num_joints
        for i in range(self.num_joints):
            qdot[i] = q_dot[i]
        joint_vel.data = qdot
        self.joint_vel_cmd_pub.publish(joint_vel)

    def pub_joint_pos(self, q):
        traj = JointTrajectory()
        # traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.joint_names
        traj_point = JointTrajectoryPoint()
        traj_point.time_from_start = rospy.Duration(3.0)
        for i in range(self.num_joints):
            traj_point.positions.append(q[i])
            traj_point.velocities.append(0.0)

        traj.points.append(traj_point)
        traj_goal = FollowJointTrajectoryGoal()
        traj_goal.trajectory = traj
        self.joint_pos_cli.send_goal(traj_goal)
        self.joint_pos_cli.wait_for_result()

    def get_eef_pose(self):
        eef_pose = kdl.Frame()
        self.pos_fk_solver.JntToCart(self.joint_state.q, eef_pose)

        return eef_pose

    def _calculate_qdot(self, eef_vel):
        eef_vel_kdl = kdl.Twist.Zero()
        eef_vel_kdl.vel[0] = eef_vel[0]
        eef_vel_kdl.vel[1] = eef_vel[1]
        eef_vel_kdl.vel[2] = eef_vel[2]
        # Transform EEF Twist to base frame
        eef_pose = kdl.Frame()
        qdot_sol = kdl.JntArray(self.num_joints)
        self.pos_fk_solver.JntToCart(self.joint_state.q, eef_pose)
        eev_vel_base = eef_pose.M * eef_vel_kdl
        self.vel_ik_solver.CartToJnt(self.joint_state.q, eev_vel_base, qdot_sol)

        return qdot_sol

    def ik(self, eef_pose):
        q_init = self.joint_state.q
        q_sol = kdl.JntArray(self.num_joints)
        self.pos_ik_solver.CartToJnt(q_init, eef_pose, q_sol)
        return q_sol

    def go_to_eef_pose(self, eef_pose):
        self.switch_controller(mode='position')
        q_sol = self.ik(eef_pose)
        self.pub_joint_pos(q_sol)

    def get_current_pixel_coords(self):
        '''Returns the 64X64 image coordinates of the gripper in the camera frame'''
        self.tf_listener.waitForTransform('camera_depth_optical_frame', 'gripper_link', rospy.Time(), rospy.Duration(0.1))
        pos, _ = self.tf_listener.lookupTransform('camera_depth_optical_frame', 'gripper_link', rospy.Time(0))
        return self.get_pixel_coords(pos)

    def get_pixel_coords(self, pos):
        '''returns pixel coordinates of the given position. Note, the position is the
        position relative to the coordinate frame of the camera'''
        #get full resolution pixel coords
        pos[1] = pos[1] / 1.282608696
        # pos[1] = pos[1] / 1.4
        x_pix, y_pix = self.cam_model.project3dToPixel(pos)
        # print(pos)
        # print('x,y : {},{}'.format(x_pix,y_pix))
        #downsample and invert pixel coords
        x_down = 63 - int(round((x_pix-self.image_offset)/7.5))
        # x_down = 63 - int(round((x_pix)/10))
        y_down = 63 - int(round(y_pix / 7.5))
        y_down = int(y_down)
        return (y_down, x_down)
        
    def get_depth_image(self):
        '''Pulls one depth image from the camera, and processes (crop, normalize
        downsample, fill in NaN values). Returns the image as a 64x64 numpy array.'''
        ima  = rospy.wait_for_message('/camera/depth/image_rect', Image, timeout = 10)
        cv_image = self.bridge.imgmsg_to_cv2(ima)
        #DEBUG uncomment to see unprocessed image
        # print('Press any key to continue...')
        # plt.imshow(cv_image)
        # plt.show()
        # cv2.imshow('image',cv_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #convert to array and set NaN to 0
        im_ar = np.nan_to_num(np.asarray(cv_image))

        #define table dimensions and boundaries
        width = im_ar.shape[1]
        height = im_ar.shape[0]
        top = 54
        bottom = 54
        left = self.image_offset
        ave_table = 1.011 #table heigh is 1.011m away from camera
        # ave_table = 1.03
        desired_table = 1.2
        desired_floor = 2.2

        #fill in zero values on table with average table height
        table = im_ar[top:-1-bottom, left:left+height]
        table[table < 0.7] = ave_table
        #fill all zero values off the table with the floor height
        floor_top = im_ar[0:top,:]
        floor_top[floor_top  < 0.1] = desired_floor
        floor_bottom = im_ar[-1-bottom:,:]
        floor_bottom[floor_bottom < 0.1] = desired_floor
        #shift the table to 1.2m from the camera
        im_ar += (desired_table - ave_table)
        #shift any remaining zeros to the table height
        im_ar[im_ar< (desired_table - ave_table)+0.1] = desired_table
        #shift everything off the table to the floor height
        im_ar[0:top,:] = desired_floor
        im_ar[-1-bottom:,:] = desired_floor
        #shift everythin off the table above a certain distance to the floor hight
        # buffer = 0.02
        # floor_top[floor_top>desired_table+buffer] = desired_floor
        # floor_bottom[floor_bottom>desired_table+buffer] = desired_floor
        #shift anything too far the floor height
        im_ar[im_ar>desired_table+0.1]=desired_floor
        #normalize & invert image
        # im_ar = 1-(im_ar-np.min(table))/(np.max(im_ar)-np.min(table))
        #crop
        im_ar_cropped = im_ar[:,left:left+height]
        # im_ar_cropped = im_ar
        # im_ar_cropped = im_ar_cropped[bottom:bottom+width,:]
        #downsample -- other interpolation options
        # interpolation = cv2.INTER_NEAREST, cv2.INTER_LANCZOS4
        im_ar_down = cv2.resize(im_ar_cropped,(64,64), interpolation = cv2.INTER_AREA)
        im_ar_down = 1 - (im_ar_down - np.min(im_ar_down)) / (np.max(im_ar_down) - np.min(im_ar_down))
        # print(im_ar_down.max(), im_ar_down.min())
        # flip the image
        im_ar_down = np.flip(im_ar_down, 0).copy()
        im_ar_down = np.flip(im_ar_down, 1).copy()
        # print(im_ar_down)
        #to see realistic view you can upsample it again to get the pixelated
        # image = cv2.resize(cv2.resize(im_ar_cropped,(64,64), interpolation = cv2.INTER_AREA),(480,480),interpolation = cv2.INTER_NEAREST)*255
        # self.show_image(im_ar_down)
        im_ar_down[:, 0:7] = 0.0
        im_ar_down[:, 64-7:64] = 0.0
        return im_ar_down

    def show_image(self, image):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(image, cmap='gray')
        plt.show()

    # def get_image(self):
    #     depth  = rospy.wait_for_message('/camera/depth/image_rect', Image, timeout = 10)
    #     depth = self.bridge.imgmsg_to_cv2(depth)
    #     depth = np.nan_to_num(np.asarray(depth))
        
    #     depth_crop = depth[:, 80:-80]
    #     # print(depth_crop.shape)
    #     table_side = 60
    #     depth_crop[0:table_side, :] = 0.0
    #     depth_crop[480-table_side:480, :] = 0.0
    #     depth_crop[:, 0:table_side] = 0.0
    #     depth_crop[:, 480-table_side:480] = 0.0
    #     depth_crop = (depth_crop - depth_crop.min()) / (depth_crop.max() - depth_crop.min())
    #     plt.imshow(depth_crop)

    #     plt.show()


def get_cv_img(info, img_size=64):
    SCALE_FACTOR = 5
    depth = info['env_map']
    belief = info['belief']
    obs_map = info['obs_map']
    true_state = info['true_state']
    timestep = info['timestep']
    pred_state = info['pred_state']
    error = np.abs(true_state - pred_state).sum()

    size = (3 * SCALE_FACTOR * img_size, SCALE_FACTOR * img_size)
    
    obs_map = ((obs_map - obs_map.min()) / (obs_map.max() - obs_map.min() + 1e-9))
    obs_map = cv2.applyColorMap(np.array(255 * obs_map, dtype=np.uint8), cv2.COLORMAP_PARULA)
    belief = cv2.applyColorMap(belief, cv2.COLORMAP_PARULA)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_BONE)
    
    if np.all(true_state < 64): 
        obs_map[true_state[0], true_state[1]] = [0, 0, 255]
        belief[true_state[0], true_state[1]] = [0, 0, 255] 
        depth[true_state[0], true_state[1]] = [0, 0, 255]
    
    image = np.hstack((depth, belief, obs_map))
    image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    cv2.putText(image, 'Timestep: {} - Error: {}'.format(timestep, error), (330, 300), font, .7, text_color, 1, cv2.LINE_AA)
    
    return image

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


if __name__ == "__main__":
    ur = URGripah()

    device = torch.device('cpu')        
    motion_model = Motion2DModel()
    obs_model = ObservationModelUNet().to(device).eval()
    obs_model_path = '/home/tarik/projects/tactile_loc/experiments/weights/unet_dataset2D_rot_enc_batch_256_best.pt'
    obs_model.load_state_dict(torch.load(obs_model_path, map_location=device))

    belief = torch.ones(64, 64) / (64 * 64)

    eef_home_pose = kdl.Frame(kdl.Rotation.Quaternion(0.0, 0.0, 0.0, 1.0), 
                              kdl.Vector(0.35, 0.43, 0.3))    
    eef_start_pose_up = kdl.Frame(kdl.Rotation.Quaternion(0.0, 0.0, 0.0, 1.0), 
                                  kdl.Vector(0.36, 0.44, 0.2))
    eef_start_pose_down = kdl.Frame(kdl.Rotation.Quaternion(0.0, 0.0, 0.0, 1.0), 
                                    kdl.Vector(0.36, 0.44, 0.12))
    
    ur.switch_controller(mode='position')

    ur.go_to_eef_pose(eef_home_pose)
    rospy.sleep(1.0)
    image = ur.get_depth_image()
    # ur.show_image(image)
    
    image_torch = torch.from_numpy(image).view(1, 1, 64, 64).to(device)

    ur.go_to_eef_pose(eef_start_pose_up)
    ur.go_to_eef_pose(eef_start_pose_down)

    print('pose : {}'.format(ur.get_current_pixel_coords()))

    ur.switch_controller(mode='velocity')
    finger_pos, states = ur.run_line_traj(image=None)

    finger_pos = running_mean(finger_pos[:, 1], 40)

    num_steps = finger_pos.shape[0]

    alpha = 0.9
    sample_freq = 16
    n_freq = 0
    finger_pos_filtered = np.zeros(num_steps, dtype=np.float32)
    finger_pos_filtered[0] = finger_pos[0]
    last_finger_pos = finger_pos[0]
    for s in range(1, num_steps):
        finger_pos_filtered[s] = alpha * (finger_pos_filtered[s - 1] + finger_pos[s] - finger_pos[s - 1])

        if ((s + 1) % sample_freq == 0):
            obs_tactile = finger_pos_filtered[n_freq * sample_freq: (n_freq + 1) * sample_freq]
            obs_tactile = torch.from_numpy(obs_tactile).unsqueeze(0).unsqueeze(0)
            if np.all(states[n_freq] < 64) and np.all(states[n_freq] >= 0):
                with torch.no_grad():
                    # MOTION UPDATE
                    pred = motion_model(belief.view(1, 1, 64, 64), 0)
                    action_torch = torch.tensor([0])

                    # OBSERVATION UPDATE
                    obs_map = obs_model(image_torch, obs_tactile, action_torch, test=True)
                
                obs_map = obs_map / obs_map.sum()
                belief = obs_map * pred
                belief = belief / belief.sum()

                # CALCULATE LOCALIZATION ERROR
                true_state = states[n_freq]
                pred_state = torch.argmax(belief).numpy()
                pred_state = np.unravel_index(pred_state, (64, 64))
                pred_state = np.array(pred_state)
                error = np.abs(true_state - pred_state).sum()

                print('True State: {}'.format(true_state))
                print('Pred State: {}'.format(pred_state))
                print('Error     : {}'.format(error))

                env_map = np.array(255 * image, dtype=np.uint8)
                belief_vis = belief.numpy()
                belief_vis = belief_vis / belief_vis.max()
                belief_vis = np.array(255 * belief_vis, dtype=np.uint8)
                info = {'true_state': true_state, 'pred_state': pred_state, 
                        'error': error, 'timestep': n_freq, 'belief': belief_vis, 'obs_map': obs_map.numpy(), 'env_map': env_map}

                vis_image = get_cv_img(info)
                cv2.imshow('Filtering', vis_image)
                cv2.waitKey(1000)

            n_freq += 1
    fig, axes = plt.subplots(3, 1, figsize=(16, 8))
    axes[0].plot(finger_pos)
    # axes[0].plot(finger_pos[:, 1])

    axes[1].plot(finger_pos_filtered)
    # axes[1].plot(finger_pos_filtered)

    plt.legend()
    plt.show()



