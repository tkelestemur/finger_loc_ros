#!/usr/bin/env python
import math
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

import numpy as np
import matplotlib.pyplot as plt

class FingerLocROS(object):
    def __init__(self):

        self.base_link = 'base_link'
        self.ee_link = 'ee_link'
        flag, self.tree = kdl_parser.treeFromParam('/robot_description')
        self.chain = self.tree.getChain(self.base_link, self.ee_link)
        self.num_joints = self.tree.getNrOfJoints()
        self.vel_ik_solver = kdl.ChainIkSolverVel_pinv(self.chain)
        self.pos_fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.pos_ik_solver = kdl.ChainIkSolverPos_LMA(self.chain)
        
        self.joint_state = kdl.JntArrayVel(self.num_joints)
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        rospy.init_node('ur_eef_vel_controller')
        rospy.Subscriber('/joint_states', JointState, self.arm_joint_state_cb)
        rospy.Subscriber('/rdda_interface/joint_states', JointState, self.finger_joint_state_cb)
        self.joint_vel_cmd_pub = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray, queue_size=10)
        self.joint_pos_cmd_pub = rospy.Publisher('/scaled_pos_traj_controller/command', JointTrajectory, queue_size=10)
        self.speed_scaling_pub = rospy.Publisher('/speed_scaling_factor', Float64, queue_size=10)
        
        
        self.switch_controller_cli = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        self.joint_pos_cli = actionlib.SimpleActionClient('/scaled_pos_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.sleep(0.5)

    def switch_controller(self, mode=None):
        req = SwitchControllerRequest()
        res = SwitchControllerResponse()
        
        req.start_asap = False
        req.timeout = 0.0
        if mode == 'velocity':
            req.start_controllers = ['joint_group_vel_controller']
            req.stop_controllers = ['scaled_pos_traj_controller']
            req.strictness = req.STRICT
        elif mode == 'position':
            req.start_controllers = ['scaled_pos_traj_controller']
            req.stop_controllers = ['joint_group_vel_controller']
            req.strictness = req.STRICT
        else:
            rospy.logwarn('Unkown mode for the controller!')

        res = self.switch_controller_cli.call(req)

        rospy.loginfo('Switched controller to ' + mode + ' : {}'.format(res.ok) )

        
    def control(self, total_time=10, eef_vel=[0, 0, 0]):
        self.switch_controller(mode='velocity')
        ctrl_freq = 128
        num_steps = int(total_time * ctrl_freq)
        rate = rospy.Rate(ctrl_freq)
        finger_pos = np.zeros((num_steps, 2))
        filter_pos = np.zeros((num_steps, 2))
        finger_vel = np.zeros((num_steps, 2))
        alpha = 0.9
        finger_start_pos = np.array(self.finger_joint_state.position)
        finger_curr_pos = np.array(self.finger_joint_state.position)
        finger_vel[0] = np.array(self.finger_joint_state.velocity)
        # finger_pos[0] = finger_curr_pos - finger_start_pos
        # filter_pos[0] = finger_curr_pos
        old_finger_pos = 0.0
        for t in range(1, num_steps):
            q_dot = self._calculate_qdot(eef_vel)
            self.pub_joint_vel(q_dot)
            finger_vel[t] = np.array(self.finger_joint_state.velocity)
            finger_curr_pos = np.array(self.finger_joint_state.position) - finger_start_pos
            finger_pos[t] = finger_curr_pos
            filter_pos[t] = alpha * (filter_pos[t-1] + finger_curr_pos - old_finger_pos)
            old_finger_pos = finger_curr_pos
            rate.sleep()
        q_zero = [0.0] * 6
        self.pub_joint_vel(q_zero)
        obs = {'finger_pos': finger_pos, 'filter_pos': filter_pos, 'finger_vel': finger_vel}
        return obs
        
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

        # print(traj)
        # self.joint_pos_cmd_pub.publish(traj)
        # rospy.sleep(traj_point.time_from_start)
        
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

    def go_to_home(self):
        self.switch_controller(mode='position')
        eef_home_pos = kdl.Vector(0.5, 0.0, 0.3)
        eef_home_rot = kdl.Rotation.Quaternion(0.0, 0.0, 0.0, 1.0)
        eef_home_pose = kdl.Frame(eef_home_rot, eef_home_pos)

        q_sol = self.ik(eef_home_pose)
        self.pub_joint_pos(q_sol)

    def go_to_start(self):
        self.switch_controller(mode='position')
        eef_start_pos = kdl.Vector(0.5, 0.41, 0.2)
        eef_start_rot = kdl.Rotation.Quaternion(0.0, 0.0, 0.0, 1.0)
        eef_start_pose = kdl.Frame(eef_start_rot, eef_start_pos)

        q_sol = self.ik(eef_start_pose)
        self.pub_joint_pos(q_sol)

        eef_start_pos = kdl.Vector(0.5, 0.41, 0.12)
        eef_start_rot = kdl.Rotation.Quaternion(0.0, 0.0, 0.0, 1.0)
        eef_start_pose = kdl.Frame(eef_start_rot, eef_start_pos)

        q_sol = self.ik(eef_start_pose)
        self.pub_joint_pos(q_sol)
        
        
def main():
    eef_vel = [0.0, -0.05, 0.0]
    finger_loc = FingerLocROS()
    finger_loc.go_to_home()
    finger_loc.go_to_start()
    obs = finger_loc.control(total_time=16, eef_vel=eef_vel)
    finger_loc.go_to_home()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(obs['finger_pos'][:, 0])
    axes[0].plot(obs['finger_pos'][:, 1])

    axes[1].plot(obs['filter_pos'][:, 0])
    axes[1].plot(obs['filter_pos'][:, 1])

    axes[2].plot(obs['finger_vel'][:, 0])
    axes[2].plot(obs['finger_vel'][:, 1])
    plt.show()


if __name__ == '__main__':
    main()
