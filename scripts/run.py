#!/usr/bin/env python
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg as geo_msgs

class FingerLocMoveIt(object):
    def __init__(self):

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('finger_loc_control')

        self.robot = moveit_commander.RobotCommander()
        self.move_group = moveit_commander.MoveGroupCommander('manipulator')
        self.move_group.set_max_velocity_scaling_factor(0.1)
        self.move_group.set_max_acceleration_scaling_factor(0.3)
        ee_pose = self.move_group.get_current_pose()
        print('Current joint values: {}'.format(ee_pose))

    def go_to_home(self):
        pose_goal = geo_msgs.Pose()
        pose_goal.orientation.w = 1.0
        pose_goal.position.x = 0.5
        pose_goal.position.y = 0.0
        pose_goal.position.z = 0.3

        self.move_group.set_pose_target(pose_goal)

        plan = self.move_group.go(wait=True)

        self.move_group.clear_pose_targets()

    def go_to_start(self):
        pose_goal = geo_msgs.Pose()
        pose_goal.orientation.w = 1.0
        pose_goal.position.x = 0.50
        pose_goal.position.y = 0.41
        pose_goal.position.z = 0.20

        self.move_group.set_pose_target(pose_goal)

        plan = self.move_group.go(wait=True)

        self.move_group.clear_pose_targets()
        self.move_group.execute(plan, wait=True)

def main():
    ctrl = FingerLocMoveIt()
    ctrl.go_to_home()

if __name__ == '__main__':
    main()
