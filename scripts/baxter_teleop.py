#!/usr/bin/env python
from IPython.nbformat import current

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Inverse Kinematics Polishing Demo
Baxter RSDK Joint Trajectory Action Client Example
"""
#!/usr/bin/env python
import argparse
import struct
import sys

from copy import copy, deepcopy
#import copy

#from future.backports.test.support import verbose

import rospy
import rospkg

import actionlib

import numpy as np
import math
import random
import time

import baxter_interface
from baxter_interface import CHECK_VERSION

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)
from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    PointStamped,
    WrenchStamped,
    Twist,
    TwistStamped,
)
from std_msgs.msg import (
    Header,
    Empty,
    UInt16,
)
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from sensor_msgs.msg import (
    Joy,
    JointState
)
from baxter_core_msgs.msg import (
    JointCommand,
    SEAJointState
)

#from baxter_tools import Tuck
from tuck_arms import Tuck
from baxter_pykdl import baxter_kinematics

from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler

global efforts
efforts = [0, 0, 0, 0, 0, 0, 0]

from pid import PID

####################################################################################

class TrajectoryControl(object):

    def __init__(self, limb, hover_distance = 0.15, sim = True, server = True, verbose = True):
        self._limb_name = limb # string
        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        #self._limb = baxter_interface.limb.Limb(limb)
        #self._arm = baxter_interface.limb.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns_ik = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns_ik, SolvePositionIK)
        rospy.wait_for_service(ns_ik, 5.0)
        
        # verify robot is enabled
        print("Getting robot state... ")
        #self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        
        print("Running. Ctrl-c to quit")
        positions = {
            'left':  [-0.11, -0.62, -1.15, 1.32,  0.80, 1.27,  2.39],
            'right':  [0.11, -0.62,  1.15, 1.32, -0.80, 1.27, -2.39],
        }
    
        if sim==True:
            # Wait for the All Clear from emulator startup
            rospy.wait_for_message("/robot/sim/started", Empty) # Comment this line to control the real Baxter (uncomment to simulated robot)

        #print("Setting up the tuck service")
        self._tuck = Tuck(True)
        self._untuck = Tuck(False)

        self._kin = baxter_kinematics(limb)
        #self._joint_names = self._arm.joint_names()
        self._joint_names = self._limb.joint_names()
        self._angles = self._limb.joint_angles()
        self._velocities = self._limb.joint_velocities()
        self._efforts = self._limb.joint_efforts()
        self._pose = self._limb.endpoint_pose()

        self._cbJoy = rospy.Subscriber("joy", Joy, self.callback_joy)
        self._cbCmdJoint = rospy.Subscriber("cmd_joint", JointState, self.callback_cmd_joint, queue_size=1)
        self._cbCmdVel = rospy.Subscriber("cmd_vel", Twist, self.callback_cmd_vel, queue_size=1)
        self._cbCmdVelStmp = rospy.Subscriber("cmd_vel/stamped", TwistStamped, self.callback_cmd_vel_stmp, queue_size=1)
        #self._cbCmdPose = rospy.Subscriber("cmd_pose", Pose, self.callback_cmd_pose, queue_size=1)
        #self._cbCmdVelPose = rospy.Subscriber("cmd_pose/stamped", PoseStamped, self.callback_cmd_pose_stmp, queue_size=1)
        self._cbCmdPose = rospy.Subscriber("cmd_pose", Twist, self.callback_cmd_pose, queue_size=1)
        self._cbCmdVelPose = rospy.Subscriber("cmd_pose/stamped", TwistStamped, self.callback_cmd_pose_stmp, queue_size=1)
        self._joy = Joy
        self._cmd_joint = JointState
        self._cmd_vel = Twist
        self._cmd_vel_stmp = TwistStamped
        #self._cmd_pose = Pose
        #self._cmd_pose_stmp = PoseStamped
        self._cmd_pose = Twist
        self._cmd_pose_stmp = TwistStamped
  
        self.q0 = 0.0 # s0
        self.q1 = 0.0 # s1
        self.q2 = 0.0 # e0
        self.q3 = 0.0 # e1
        self.q4 = 0.0 # w0
        self.q5 = 0.0 # w1
        self.q6 = 0.0 # w2
        
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.wx = 0.0
        self.wy = 0.0
        self.wz = 0.0
        
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        self.dR = 0.0
        self.dP = 0.0
        self.dY = 0.0
        
        self.ex = 0.0
        self.ey = 0.0
        self.ez = 0.0
        self.eR = 0.0
        self.eP = 0.0
        self.eY = 0.0
        
        self.ux = 0.0
        self.uy = 0.0
        self.uz = 0.0
        self.uR = 0.0
        self.uP = 0.0
        self.uY = 0.0

        # Values for simulated Baxter robot
        self.vx_pid = PID(2.0, 0.0, 0.1, 1.0, -1.0)
        self.vy_pid = PID(2.0, 0.0, 0.1, 1.0, -1.0)
        self.vz_pid = PID(2.0, 0.0, 0.1, 1.0, -1.0)
        self.wx_pid = PID(1.0, 0.0, 0.1, 1.0, -1.0)
        self.wy_pid = PID(1.0, 0.0, 0.1, 1.0, -1.0)
        self.wz_pid = PID(1.0, 0.0, 0.1, 1.0, -1.0)
        
#         # Values for real Baxter robot
#         self.vx_pid = PID(10.0, 0.0, 0.1, 1.0, -1.0)
#         self.vy_pid = PID(10.0, 0.0, 0.1, 1.0, -1.0)
#         self.vz_pid = PID(10.0, 0.0, 0.1, 1.0, -1.0)
#         self.wx_pid = PID(5.0, 0.0, 1.0, 1.0, -1.0)
#         self.wy_pid = PID(5.0, 0.0, 1.0, 1.0, -1.0)
#         self.wz_pid = PID(5.0, 0.0, 1.0, 1.0, -1.0)
        
        # Set new_cmd_vel, new_cmd_pose, new_cmd_joint as false
        self.new_cmd_vel = False
        self.new_cmd_pose = False
        self.new_cmd_joint = False
        self.record_pose = True
        
        new_pose_msg = Pose()
        
        ns_grav = "/robot/limb/" + limb + "/gravity_compensation_torques"
        self._cbGravity = rospy.Subscriber(ns_grav, SEAJointState, self.callback_gravity)
        self._gravity = SEAJointState
        
        self._cmd = JointCommand()
        ns_cmd = "robot/limb/" + limb + "/joint_command"
        self._pub_cmd = rospy.Publisher(ns_cmd,
                                         JointCommand, queue_size=10)
        
        """
        'Velocity control' of one of the arms by commanding joint velocities.
        (based on Cartesian velocity and Jacobian???).
        """
        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
                                         UInt16, queue_size=10)
        # control parameters
        self._rate = 500.0  # Hz
        # set joint state publishing to 500Hz
        self._pub_rate.publish(self._rate)    
        
        """
        'Trajectory control' of one of the arms by sending trajectories to
        the action server.
        """
        self._server = server
        if server:
            ns_traj = 'robot/limb/' + limb + '/'
            self._client = actionlib.SimpleActionClient(
                ns_traj + "follow_joint_trajectory",
                FollowJointTrajectoryAction,
            )
            self._goal = FollowJointTrajectoryGoal()
            self._goal_time_tolerance = rospy.Time(0.1)
            self._goal.goal_time_tolerance = self._goal_time_tolerance
            server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
            if not server_up:
                rospy.logerr("Timed out waiting for Joint Trajectory"
                             " Action Server to connect. Start the action server"
                             " before running example.")
                rospy.signal_shutdown("Timed out waiting for Action Server")
                sys.exit(1)
            self.clear(limb)
    
    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if 'right_measure_joint' in limb_joints:
                del limb_joints['right_measure_joint']
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles, timeout=10.0):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles, timeout)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def move_to_neutral(self, timeout=10.0):
        print("Moving the {0} arm to neutral pose...".format(self._limb_name))
        self._limb.move_to_neutral(timeout)
        #self._arm.move_to_neutral() #Sets arm back into a neutral pose.
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def move_to_angles(self, start_angles=None, timeout=10.0):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles, timeout)
        print("Running. Ctrl-c to quit")

    def move_to_pose(self, pose, timeout=5.0):
        print pose
        # servo down to release
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles, timeout)

    def set_joint_position(self, joint_positions):
        if joint_positions:
            self._limb.set_joint_positions(joint_positions)
            #self._arm.set_joint_positions(joint_positions)
        else:
            rospy.logerr("No Joint Position provided for set_joint_positions. Staying put.")

    def set_joint_velocity(self, joint_velocities):
        if joint_velocities:
            self._limb.set_joint_velocities(joint_velocities)
            #self._arm.set_joint_velocities(joint_velocities)
        else:
            rospy.logerr("No Joint Velocity provided for set_joint_velocities. Staying put.")

    def set_joint_torque(self, joint_torques):
        if joint_torques:
            self._limb.set_joint_torques(joint_torques)
        else:
            rospy.logerr("No Joint Torque provided for set_joint_torques. Staying put.")

    def gripper_calibrate(self):
        self._gripper.calibrate()
        rospy.sleep(5.0)
        
    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(1.0)

    def _approach(self, pose, hover_distance=0.1, timeout=2.0):
        approach = deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + hover_distance
        joint_angles = self.ik_request(approach)
        self._guarded_move_to_joint_position(joint_angles, timeout)

    def _retract(self, timeout=2.0):
        # retrieve current pose from endpoint
        pose = self._limb.endpoint_pose()
        pose_msg = self.get_pose_msg(pose)
        joint_angles = self.get_angles(pose_msg)
        # servo up from current pose
        self._guarded_move_to_joint_position(joint_angles, timeout)

    def get_pose_msg(self, pose):
        # convert to pose msg
        pose_msg = Pose()
        pose_msg.position.x = pose['position'].x
        pose_msg.position.y = pose['position'].y
        pose_msg.position.z = pose['position'].z
        pose_msg.orientation.x = pose['orientation'].x
        pose_msg.orientation.y = pose['orientation'].y
        pose_msg.orientation.z = pose['orientation'].z
        pose_msg.orientation.w = pose['orientation'].w
        return pose_msg

    def get_angles(self, pose, hover_distance=0.0):
        # get joint angles from pose        
        pose_hover = deepcopy(pose)
        pose_hover.position.z = pose_hover.position.z + hover_distance # Get angles for the position + hover distance
        joint_angles = self.ik_request(pose_hover)
#         position = [pose.position.x, pose.position.y, pose.position.z]
#         orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w] # Info: http://docs.ros.org/diamondback/api/kdl/html/python/geometric_primitives.html#rotation
#         joint_angles = self._kin.inverse_kinematics(position, orientation) # More info: https://pypi.org/project/PyKDL/
        return joint_angles

    def get_angles_array(self, pose, hover_distance=0.0):
        # get joint angles from pose        
        pose.position.z = pose.position.z + hover_distance # Get angles for the position + hover distance
        joint_angles = self.ik_request(pose)
        #joint_angles = self.get_angles(pose, hover_distance)
        
        if joint_angles:
            joint_angles_array = joint_angles.values()
            joint_angles_array[0] = joint_angles['right_s0']
            joint_angles_array[1] = joint_angles['right_s1']
            joint_angles_array[2] = joint_angles['right_e0']
            joint_angles_array[3] = joint_angles['right_e1']
            joint_angles_array[4] = joint_angles['right_w0']
            joint_angles_array[5] = joint_angles['right_w1']
            joint_angles_array[6] = joint_angles['right_w2']
#             position = [pose.position.x, pose.position.y, pose.position.z]
#             orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w] # Info: http://docs.ros.org/diamondback/api/kdl/html/python/geometric_primitives.html#rotation
#             joint_angles = self._kin.inverse_kinematics(position, orientation) # More info: https://pypi.org/project/PyKDL/
            return joint_angles_array
        else:
            return joint_angles
        
    def pick(self, pose):
        # open the gripper
        self.gripper_open()
        # servo above pose
        self._approach(pose, timeout=5.0)
        # servo to pose
        self.move_to_pose(pose)
        # close gripper
        self.gripper_close()
        # retract to clear object
        self._retract()

    def place(self, pose):
        # servo above pose
        self._approach(pose, timeout=5.0)
        # servo to pose
        self.move_to_pose(pose)
        # open the gripper
        self.gripper_open()
        # retract to clear object
        self._retract()

    def generate_trajectory(self, initial, final, pts=10):
        trajectory = []
        for i in range(0, pts):
            new_pose = Pose()
            new_pose.position.x = initial.position.x+(final.position.x-initial.position.x)*i/pts
            new_pose.position.y = initial.position.y+(final.position.y-initial.position.y)*i/pts
            new_pose.position.z = initial.position.z+(final.position.z-initial.position.z)*i/pts
            new_pose.orientation = initial.orientation
            trajectory.append(new_pose)
            #polish_poses.append(Pose(
            #position=Point(x, y, z),
            #orientation=overhead_orientation))
        return trajectory
    
    def angle_diff(self, angle1, angle2):
        
         # Rotate angle1 with angle2 so that the sought after
         # angle is between the resulting angle and the x-axis
         angle = angle1 - angle2

         # "Normalize" angle to range [-180,180[
         if angle < -math.pi:
             angle = angle+math.pi*2
         elif angle > math.pi:
             angle = angle-math.pi*2
        
         return angle
        
    def callback_joy(self, data):
        self._joy = data
        #print(self._joy)
        
    def callback_cmd_joint(self, data):
        self._cmd_joint = data
        #print(self._cmd_joint)
        # Set angular velocities
        self.q0 = self._cmd_joint.velocity[0] # s0
        self.q1 = self._cmd_joint.velocity[1] # s1
        self.q2 = self._cmd_joint.velocity[2] # e0
        self.q3 = self._cmd_joint.velocity[3] # e1
        self.q4 = self._cmd_joint.velocity[4] # w0
        self.q5 = self._cmd_joint.velocity[5] # w1
        self.q6 = self._cmd_joint.velocity[6] # w2
        # Reset cartesian velocities
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.wx = 0.0
        self.wy = 0.0
        self.wz = 0.0
        # Reset cartesian pose
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        self.dR = 0.0
        self.dP = 0.0
        self.dY = 0.0
        # Set new_cmd_joint as true
        #if (self.q0!=0.0 or self.q1!=0.0 or self.q2!=0.0 or self.q3!=0.0 or self.q4!=0.0 or self.q5!=0.0 or self.q6!=0.0):
        if (math.fabs(self.q0)>0.01 or math.fabs(self.q1)>0.01 or math.fabs(self.q2)>0.01 or math.fabs(self.q3)>0.01 or math.fabs(self.q4)>0.01 or math.fabs(self.q5)>0.01 or math.fabs(self.q6)>0.01):
            self.new_cmd_joint = True
            self.record_pose = True
        
    def callback_cmd_vel(self, data):
        self._cmd_vel = data
        #print(self._cmd_vel)
        
    def callback_cmd_vel_stmp(self, data):
        self._cmd_vel_stmp = data
        #print(self._cmd_vel_stmp)
        # Set cartesian velocities
        self.vx = self._cmd_vel_stmp.twist.linear.x
        self.vy = self._cmd_vel_stmp.twist.linear.y
        self.vz = self._cmd_vel_stmp.twist.linear.z
        self.wx = self._cmd_vel_stmp.twist.angular.x
        self.wy = self._cmd_vel_stmp.twist.angular.y
        self.wz = self._cmd_vel_stmp.twist.angular.z
        # Reset angular velocities
        self.q0 = 0.0 # s0
        self.q1 = 0.0 # s1
        self.q2 = 0.0 # e0
        self.q3 = 0.0 # e1
        self.q4 = 0.0 # w0
        self.q5 = 0.0 # w1
        self.q6 = 0.0 # w2
        # Reset cartesian pose
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        self.dR = 0.0
        self.dP = 0.0
        self.dY = 0.0
        # Set new_cmd_vel as true
        #if (self.vx!=0.0 or self.vy!=0.0 or self.vz!=0.0 or self.wx!=0.0 or self.wy!=0.0 or self.wz!=0.0):
        if (math.fabs(self.vx)>0.01 or math.fabs(self.vy)>0.01 or math.fabs(self.vz)>0.01 or math.fabs(self.wx)>0.01 or math.fabs(self.wy)>0.01 or math.fabs(self.wz)>0.01):
            self.new_cmd_vel = True
            self.record_pose = True
        
    def callback_cmd_pose(self, data):
        self._cmd_pose = data
        #print(self._cmd_pose)
        
    def callback_cmd_pose_stmp(self, data):
        self._cmd_pose_stmp = data
        #print(self._cmd_pose_stmp)
        
        if(self.record_pose == True):
  
            # The Pose of the movement in its initial location.
            current_pose_msg = self.get_pose_msg(self._limb.endpoint_pose())
            
            self.initial_position = [current_pose_msg.position.x,
                                     current_pose_msg.position.y,
                                     current_pose_msg.position.z]

#             self.initial_quaternion = [current_pose_msg.orientation.w, 
#                                        current_pose_msg.orientation.x, 
#                                        current_pose_msg.orientation.y, 
#                                        current_pose_msg.orientation.z]
#             
#             euler = euler_from_quaternion(self.initial_quaternion)
#             print euler
            
            # An orientation for gripper fingers to be overhead and parallel to the obj
            roll = 0
            pitch = 0
            yaw = math.pi
            self.initial_quaternion = quaternion_from_euler(roll, pitch, yaw)

            self.initial_euler = euler_from_quaternion(self.initial_quaternion)
            
            self.initial_joints = self._limb.joint_angles()
            
            self.record_pose = False
            
#         # Set cartesian pose (position + orientation)
#         self.dx = self._cmd_pose_stmp.pose.position.x
#         self.dy = self._cmd_pose_stmp.pose.position.y
#         self.dz = self._cmd_pose_stmp.pose.position.z
#         
#         delta_quaternion = [self._cmd_pose_stmp.pose.orientation.w, 
#                             self._cmd_pose_stmp.pose.orientation.x, 
#                             self._cmd_pose_stmp.pose.orientation.y, 
#                             self._cmd_pose_stmp.pose.orientation.z]
#                    
#         delta_euler = euler_from_quaternion(delta_quaternion)
#         
#         self.dR = delta_euler[2]
#         self.dP = delta_euler[1]
#         self.dY = delta_euler[0]

        # Set cartesian pose (position + orientation)
        self.dx = self._cmd_pose_stmp.twist.linear.x
        self.dy = self._cmd_pose_stmp.twist.linear.y
        self.dz = self._cmd_pose_stmp.twist.linear.z
        self.dR = self._cmd_pose_stmp.twist.angular.x
        self.dP = self._cmd_pose_stmp.twist.angular.y
        self.dY = self._cmd_pose_stmp.twist.angular.z

        self.new_x = self.initial_position[0] + self.dx
        self.new_y = self.initial_position[1] + self.dy
        self.new_z = self.initial_position[2] + self.dz
        new_point = Point(x=self.new_x, y=self.new_y, z=self.new_z)
     
        self.new_Y = self.initial_euler[0] + self.dY
        self.new_P = self.initial_euler[1] + self.dP
        self.new_R = self.initial_euler[2] + self.dR
        new_quaterion = quaternion_from_euler(self.new_R, self.new_P, self.new_Y)
       
        new_orientation = Quaternion(x=new_quaterion[1],
                                     y=new_quaterion[2],
                                     z=new_quaterion[3],
                                     w=new_quaterion[0])

        self.new_pose_msg = Pose(position=new_point,
                                 orientation=new_orientation)

        # Reset cartesian velocities
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.wx = 0.0
        self.wy = 0.0
        self.wz = 0.0
        # Reset angular velocities
        self.q0 = 0.0 # s0
        self.q1 = 0.0 # s1
        self.q2 = 0.0 # e0
        self.q3 = 0.0 # e1
        self.q4 = 0.0 # w0
        self.q5 = 0.0 # w1
        self.q6 = 0.0 # w2
        # Set new_cmd_vel as true
        #if (self.dx!=0.0 or self.dy!=0.0 or self.dz!=0.0 or self.dR!=0.0 or self.dP!=0.0 or self.dY!=0.0):
        if (math.fabs(self.dx)>0.001 or math.fabs(self.dy)>0.001 or math.fabs(self.dz)>0.001 or math.fabs(self.dR)>0.01 or math.fabs(self.dP)>0.01 or math.fabs(self.dY)>0.01):
            self.new_cmd_pose = True
        
    def callback_gravity(self, data):
        self._gravity = data
        global efforts
        #efforts = np.negative(self._gravity.gravity_model_effort)
        efforts = self._gravity.gravity_model_effort

    """
    'Velocity control' of one of the arms by commanding joint velocities.
    (based on Cartesian velocity and Jacobian???).
    """

    def wobble(self):
        self.move_to_neutral()
        """
        Performs the wobbling of both arms.
        """
        rate = rospy.Rate(self._rate)
        start = rospy.Time.now()
        
        def make_v_func():
            """
            returns a randomly parameterized cosine function to control a
            specific joint.
            """
            period_factor = random.uniform(0.3, 0.5)
            amplitude_factor = random.uniform(0.1, 0.2)

            def v_func(elapsed):
                w = period_factor * elapsed.to_sec()
                return amplitude_factor * math.cos(w * 2 * math.pi)
            return v_func

        v_funcs = [make_v_func() for _ in self._joint_names]

        def make_cmd(joint_names, elapsed):
            return dict([(joint, v_funcs[i](elapsed))
                         for i, joint in enumerate(joint_names)])
            
#         def make_cmd(self, joint_names, elapsed):
#             
# #             jacobian = self._kin.jacobian()
# #             jacobianInv = self._kin.jacobian_pseudo_inverse()
# #     
# #             u_cartesian = np.array([[0], [0], [0], [0], [0], [0]]) # vx, vy, vz, wx, wy, wz
# #         
# #             u_joints = jacobianInv * u_cartesian
# #             
# #             u_joints = np.array([[0.5], [0], [0], [0], [0], [0], [0]]) # u0, u1, u2, u3, u4, u5, u6
# #     
# #             #joint_velocities = dict(zip(traj._arm._joint_names['right'],
# #             #                  u_joints))
# #     
# #             joint_velocities = dict(zip(joint_names,
# #                               u_joints))
#             joint_velocities = dict([(joint, u_joints)
#                         for i, joint in enumerate(joint_names)])
#             
#             print joint_names
#             
#             return joint_velocities
#             
#             #return dict([(joint, v_funcs[i](elapsed))
#             #             for i, joint in enumerate(joint_names)])

        print("Press Ctrl-C to stop...")
        while not rospy.is_shutdown():
            self._pub_rate.publish(self._rate)
            elapsed = rospy.Time.now() - start
            cmd = make_cmd(self._joint_names, elapsed)
            self._limb.set_joint_velocities(cmd)
            #self._arm.set_joint_velocities(cmd)
            rate.sleep()

#         while not rospy.is_shutdown():
#             self._pub_rate.publish(self._rate)
#             elapsed = rospy.Time.now() - start
#             #cmd = make_cmd(self._left_joint_names, elapsed)
#             #self._left_arm.set_joint_velocities(cmd)
# #             cmd = make_cmd(self._joint_names, elapsed)
# #             cmd[self._joint_names[0]]=-0.2
# #             self._arm.set_joint_velocities(cmd)
# #             self._cmd.mode = self._cmd.POSITION_MODE
# #             self._cmd.command = self._angles
#             self._cmd.mode = self._cmd.VELOCITY_MODE
#             self._cmd.command = [0,0,0.2,0,0,0,0]
#             self._cmd.names = self._joint_names
#             print self._cmd.command
#             self._pub_cmd.publish(self._cmd)
#             rate.sleep()

####################################################################################
            
    def pick_place(self):
        idx = 0
        while not rospy.is_shutdown():
            print("\nPicking...")
            traj.pick(block_poses[idx])
            print("\nPlacing...")
            idx = (idx+1) % len(block_poses)
            traj.place(block_poses[idx])

####################################################################################

    def loop(self, fc = 100.0):
        
        # Joint angles for right arm (initial position to avoid table)
        joint_angles = {'right_s0': -0.5,
                        'right_s1': -1.0,
                        'right_e0': 0.0,
                        'right_e1': 2.0,
                        'right_w0': 0.0,
                        'right_w1': 1.0,
                        'right_w2': 0.0}
        self.move_to_angles(joint_angles, timeout=3.0)
        
        # Joint angles for right arm (tuck position)
        joint_angles = {'right_s0': 0,
                        'right_s1': -1.0,
                        'right_e0': 1.2,
                        'right_e1': 2.0,
                        'right_w0': -0.6,
                        'right_w1': 1.0,
                        'right_w2': 0.5}
        self.move_to_angles(joint_angles, timeout=3.0)
        
        # Joint angles for right arm (zero in all joints)
        joint_angles = {'right_s0': 0.0,
                        'right_s1': 0.0,
                        'right_e0': 0.0,
                        'right_e1': 0.0,
                        'right_w0': 0.0,
                        'right_w1': 0.0,
                        'right_w2': 0.0}
        self.move_to_angles(joint_angles, timeout=3.0)

        tc = 1.0/fc
        rate = rospy.Rate(fc)
        #rospy.spin()
        
        joint_positions = dict()
        joint_velocities = dict()
        
        # Enable Baxter again??
        #baxter_interface.RobotEnable(CHECK_VERSION)
        
#         current_pose = self._pose # Save initial pose
#         current_position = current_pose['position']
#         current_orientation = current_pose['orientation']
#             
#         # The Pose of the movement in its initial location.
#         current_pose_msg = Pose()
#         current_pose_msg = self.get_pose_msg(current_pose)
        
        # Save current joint angles
#         u_joints = self.get_angles_array(current_pose_msg, hover_distance=0.0) # This is very noisy because of IK
#         print u_joints
#         u_joints = self._limb.joint_angles().values()
#         print u_joints
        joint_positions = self._limb.joint_angles()
        
        while not rospy.is_shutdown():
            
#             if self.new_cmd_vel:
#                   
#                 self.new_cmd_vel = False
#                   
#                 new_x = current_pose_msg.position.x + self.vx*tc
#                 new_y = current_pose_msg.position.y + self.vy*tc
#                 new_z = current_pose_msg.position.z + self.vz*tc
#                 new_point = Point(x=new_x, y=new_y, z=new_z)
#                   
#                 current_quaternion = [current_pose_msg.orientation.w, 
#                                       current_pose_msg.orientation.x, 
#                                       current_pose_msg.orientation.y, 
#                                       current_pose_msg.orientation.z]
#                   
#                 current_euler = euler_from_quaternion(current_quaternion)
#       
#                 new_Y = current_euler[0] + self.wz*tc
#                 new_P = current_euler[1] + self.wy*tc
#                 new_R = current_euler[2] + self.wx*tc
#                   
#                 new_quaterion = quaternion_from_euler(new_R, new_P, new_Y)
#       
#                 new_orientation = Quaternion(x=-new_quaterion[1],
#                                              y=-new_quaterion[2],
#                                              z=-new_quaterion[3],
#                                              w=-new_quaterion[0])
#                   
#                 new_pose_msg = Pose(position=new_point,
#                                     orientation=new_orientation)
#                   
#                 #self.move_to_pose(new_pose_msg, timeout=tc)
#                 #rospy.sleep(tc)
#       
#                 # servo down to release
#                 #joint_angles = self.ik_request(new_pose_msg)
#                 #self._guarded_move_to_joint_position(joint_angles, timeout=tc)
#                   
#                   
#                 #print self.vx*tc, " ", self.vy*tc, " ", self.vz*tc
#                 #print current_pose_msg
#                 #print new_pose_msg
#                   
#                 #if(new_pose_msg.position!=current_pose_msg.position or new_pose_msg.orientation!=current_pose_msg.orientation):
#                 u_joints = self.get_angles_array(new_pose_msg, hover_distance=0.0)
#                 if(u_joints):
#                     joint_positions = dict(zip(self._joint_names, u_joints))
#                     #self.set_joint_position(joint_positions)
#                     self._limb.move_to_joint_positions(joint_positions, timeout=tc)
#                     # Save new pose message for new iteration
#                     current_pose_msg = new_pose_msg

#######################################################################

            # Joint angles/velocities depending on the controller used
            #u_joints = np.array([[self.q0], [self.q1], [self.q2], [self.q3], [self.q4], [self.q5], [self.q6]]) # Define a column vector?
            u_joints = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) # Define a column vector?
            #u_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Define a column vector?
              
            if self.new_cmd_vel:
                   
                self.new_cmd_vel = False
                # Cartesian velocities (linear and angular)
                u_cartesian = np.array([[self.vx], [self.vy], [self.vz], [self.wx], [self.wy], [self.wz]]) # Define a column vector
                # Computation of joint velocities from cartesian space using Jacobian matrix
                jacobianInv = self._kin.jacobian_pseudo_inverse()
                u_joints = jacobianInv * u_cartesian
                
            elif self.new_cmd_joint:
                   
                self.new_cmd_joint = False
                # Joint angles/velocities depending on the controller used
                u_joints = np.array([[self.q0], [self.q1], [self.q2], [self.q3], [self.q4], [self.q5], [self.q6]]) # Define a column vector

            elif self.new_cmd_pose:
                   
                self.new_cmd_pose = False
               
#                 #self.move_to_pose(self.new_pose_msg, timeout=tc)
# 
#                 #if(new_pose_msg.position!=current_pose_msg.position or new_pose_msg.orientation!=current_pose_msg.orientation):
#                 u_joints = self.get_angles_array(self.new_pose_msg, hover_distance=0.0)
#                 if(u_joints):
#                     joint_positions = dict(zip(self._joint_names, u_joints))
#                     #self.set_joint_position(joint_positions)
#                     self._limb.move_to_joint_positions(joint_positions, timeout=tc)
#                     # Save new pose message for new iteration
#                     #current_pose_msg = new_pose_msg

                current_pose_msg = self.get_pose_msg(self._limb.endpoint_pose())
 
                current_quaternion = [current_pose_msg.orientation.w, 
                                      current_pose_msg.orientation.x, 
                                      current_pose_msg.orientation.y, 
                                      current_pose_msg.orientation.z]
                    
                current_euler = euler_from_quaternion(current_quaternion)
                 
                self.ex = self.new_x - current_pose_msg.position.x
                self.ey = self.new_y - current_pose_msg.position.y
                self.ez = self.new_z - current_pose_msg.position.z
                #self.eR = self.new_R - current_euler[2]
                #self.eP = self.new_P - current_euler[1]
                #self.eY = self.new_Y - current_euler[0]
                self.eR = self.angle_diff(self.new_R, current_euler[2])
                self.eP = self.angle_diff(self.new_P, current_euler[1])
                self.eY = self.angle_diff(self.new_Y, current_euler[0])
                
                self.ux = -self.vx_pid.update_PID(self.ex)
                self.uy = -self.vy_pid.update_PID(self.ey)
                self.uz = -self.vz_pid.update_PID(self.ez)
                self.uR = self.wx_pid.update_PID(self.eR)
                self.uP = -self.wy_pid.update_PID(self.eP)
                self.uY = self.wz_pid.update_PID(self.eY)

                # Cartesian velocities (linear and angular)
                u_cartesian = np.array([[self.ux], [self.uy], [self.uz], [self.uR], [self.uP], [self.uY]]) # Define a column vector
                # Computation of joint velocities from cartesian space using Jacobian matrix
                jacobianInv = self._kin.jacobian_pseudo_inverse()
                u_joints = jacobianInv * u_cartesian

            else:
                   
                self.new_cmd_vel = False
                self.new_cmd_joint = False
                self.new_cmd_pose = False
                #self.record_pose = True
                # Joint angles/velocities depending on the controller used
                u_joints = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) # Define a column vector

#######################################################################

            # Update joint_positions only if there's a big difference between actual and reference joint positions (to avoid stuck configurations)
            joint_positions_current = self._limb.joint_angles()
            error_norm = np.linalg.norm(np.array(joint_positions_current.values()).transpose()-np.array(joint_positions.values()).transpose())
            if(error_norm>0.05):
                joint_positions = joint_positions_current

#######################################################################
            
            #u_joints = self.get_angles_array(current_pose_msg, hover_distance=0.0) # This is very noisy because of IK
            u_joints[0] = joint_positions['right_s0'] + u_joints[0]*tc
            u_joints[1] = joint_positions['right_s1'] + u_joints[1]*tc
            u_joints[2] = joint_positions['right_e0'] + u_joints[2]*tc
            u_joints[3] = joint_positions['right_e1'] + u_joints[3]*tc
            u_joints[4] = joint_positions['right_w0'] + u_joints[4]*tc
            u_joints[5] = joint_positions['right_w1'] + u_joints[5]*tc
            u_joints[6] = joint_positions['right_w2'] + u_joints[6]*tc
                
#######################################################################
 
            ## POSITION CONTROL
            #joint_positions = dict(zip(self._limb._joint_names[self._limb_name], u_joints))
            joint_positions = dict(zip(self._joint_names, u_joints)) #u_joints = self.get_angles_array(current_pose_msg, hover_distance=0.0) # This is very noisy because of IK
   
            ## APPLY POSITION CONTROL ACTION
            #self.move_to_angles(start_angles=joint_positions, timeout=tc) # Not good inside a control loop
            self.set_joint_position(joint_positions)

#######################################################################

#             ## VELOCITY CONTROL
#             #joint_velocities = dict(zip(self._limb._joint_names[self._limb_name], u_joints))
#             joint_velocities = dict(zip(self._joint_names, u_joints))
#               
#             ## APPLY VELOCITY CONTROL ACTION
#             self.set_joint_velocity(joint_velocities)
 
#######################################################################

#             ## EFFORT CONTROL
#             #joint_efforts = dict(zip(self._limb._joint_names[self._limb_name], u_joints))
#             joint_efforts = dict(zip(self._joint_names, efforts))
#  
#             ## APPLY EFFORT CONTROL ACTION
#             #self.set_joint_torque(joint_efforts)
 
#######################################################################

#             # Control of Baxter robot by publishing a command with values and type of control
#             # This is equivalent to set_joint_position(joint_positions) and set_joint_velocity(joint_velocities)
#             ## POSITION CONTROL
#             self._cmd.mode = self._cmd.POSITION_MODE
#             self._cmd.names = joint_positions.keys()
#             self._cmd.command = joint_positions.values()
#
#             ## VELOCITY CONTROL
#             self._cmd.mode = self._cmd.VELOCITY_MODE
#             self._cmd.names = joint_velocities.keys()
#             self._cmd.command = joint_velocities.values()
#
#             # PUBLISH CMD CONTROL
#             self._pub_cmd.publish(self._cmd)

#######################################################################
            #rospy.sleep(tc)
            rate.sleep()
            #print idx
            
#######################################################################

    def action(self):
        
        # An orientation for gripper fingers to be overhead and parallel to the obj
        roll = -math.pi
        pitch = math.pi
        yaw = 0.0
        quaterion = quaternion_from_euler(roll, pitch, yaw)
        #print overhead_orientation
        overhead_orientation = Quaternion(x=quaterion[1],
                                          y=quaterion[2],
                                          z=quaterion[3],
                                          w=quaterion[0])
     
        # Starting angles for right arm
        start_angles1 = {'right_s0': 0,
                        'right_s1': 0,
                        'right_e0': 0,
                        'right_e1': 0,
                        'right_w0': 0,
                        'right_w1': 0,
                        'right_w2': 0}
        
        # Starting angles for right arm
        start_angles2 = {'right_s0': 0.5,
                        'right_s1': 0,
                        'right_e0': 0,
                        'right_e1': 0,
                        'right_w0': 0,
                        'right_w1': 0,
                        'right_w2': 0}
        
        # Starting angles for right arm
        start_angles3 = {'right_s0': 0.5,
                        'right_s1': 0.5,
                        'right_e0': 0,
                        'right_e1': 0,
                        'right_w0': 0,
                        'right_w1': 0,
                        'right_w2': 0}
        
        # Starting angles for right arm
        start_angles4 = {'right_s0': 0.5,
                        'right_s1': 0.5,
                        'right_e0': 0.5,
                        'right_e1': 0,
                        'right_w0': 0,
                        'right_w1': 0,
                        'right_w2': 0}
        
        # Starting angles for right arm
        start_angles5 = {'right_s0': 0.5,
                        'right_s1': 0.5,
                        'right_e0': 0.5,
                        'right_e1': 0.5,
                        'right_w0': 0,
                        'right_w1': 0,
                        'right_w2': 0}
        
        # Starting angles for right arm
        start_angles6 = {'right_s0': 0.5,
                        'right_s1': 0.5,
                        'right_e0': 0.5,
                        'right_e1': 0.5,
                        'right_w0': 0.5,
                        'right_w1': 0,
                        'right_w2': 0}
        
        # Starting angles for right arm
        start_angles7 = {'right_s0': 0.5,
                        'right_s1': 0.5,
                        'right_e0': 0.5,
                        'right_e1': 0.5,
                        'right_w0': 0.5,
                        'right_w1': 0.5,
                        'right_w2': 0}
        
        # Starting angles for right arm
        start_angles8 = {'right_s0': 0.5,
                        'right_s1': 0.5,
                        'right_e0': 0.5,
                        'right_e1': 0.5,
                        'right_w0': 0.5,
                        'right_w1': 0.5,
                        'right_w2': 0.5}
        
        # Joint angles for right arm
        joint_angles = {'right_s0': 0,
                        'right_s1': -1.0,
                        'right_e0': 1.2,
                        'right_e1': 2.0,
                        'right_w0': -0.6,
                        'right_w1': 1.0,
                        'right_w2': 0.5}
           
        # First pose
        pose00 = Pose(position=Point(x=0.5, y=0.0, z=0.1),
                      orientation=overhead_orientation)
        # Second pose
        pose01 = Pose(position=Point(x=0.5, y=-0.2, z=0.1),
                      orientation=overhead_orientation)
        # Untuck pose (position and orientation)
        pose02 = Pose(position=Point(x=0.5, y=-0.3, z=0.1),
                     orientation=overhead_orientation)
        
        # The Pose of the block in its initial location.
        # You may wish to replace these poses with estimates
        # from a perception node.
        # Feel free to add additional desired poses for the object.
        # Each additional pose will get its own pick and place.
        block_poses = list()
        pose_block_00 = Pose(position=Point(x=0.7, y=0.2, z=-0.2),
                            orientation=overhead_orientation)
        block_poses.append(pose_block_00)
        pose_block_01 = Pose(position=Point(x=0.7, y=0.0, z=-0.2),
                             orientation=overhead_orientation)
        block_poses.append(pose_block_01)
        pose_block_02 = Pose(position=Point(x=0.7, y=-0.2, z=-0.2),
                            orientation=overhead_orientation)
        block_poses.append(pose_block_02)

#         # Move to the desired starting angles
#         self.move_to_angles(start_angles1, timeout=3.0)
#         # Move to the desired starting angles
#         self.move_to_angles(start_angles2, timeout=3.0)
#         # Move to the desired starting angles
#         self.move_to_angles(start_angles3, timeout=3.0)
#         # Move to the desired starting angles
#         self.move_to_angles(start_angles4, timeout=3.0)
#         # Move to the desired starting angles
#         self.move_to_angles(start_angles5, timeout=3.0)
#         # Move to the desired starting angles
#         self.move_to_angles(start_angles6, timeout=3.0)
#         # Move to the desired starting angles
#         self.move_to_angles(start_angles7, timeout=3.0)
#         # Move to the desired starting angles
#         self.move_to_angles(start_angles8, timeout=3.0)
        # Move to the desired starting angles
        self.move_to_angles(joint_angles, timeout=3.0)
     
        #new_angles = dict(zip(self._joint_names, joint_angles.values()))
        #print new_angles
        #print joint_angles
         
        # Move to a new cartesian pose
        #self.move_to_pose(block_poses[-1], timeout=3.0)
        self.move_to_pose(pose02, timeout=3.0)
        rospy.sleep(1.0)
     
        # Open gripper
        self.gripper_open()
        rospy.sleep(1.0)
     
        # Close gripper
        self.gripper_close()
        rospy.sleep(1.0)
 
##########################################################################################
     
        # The Pose of the movement in its initial location.
        pose03 = self.get_pose_msg(self._pose)
        pose04 = deepcopy(pose03)
        pose04.position.z = pose04.position.z-0.2;
        polish_poses = []
        polish_poses = self.generate_trajectory(pose03,pose04,100)
        polish_poses.extend(self.generate_trajectory(pose04,pose03,100))
    
#         current_pose = self._pose
#         # The Pose of the movement in its initial location.
#         pose1 = Pose()
#         pose1 = self.get_pose_msg(current_pose)
#         pose2 = Pose()
#         pose2 = deepcopy(pose1)
#         pose2.position.z = pose1.position.z-0.5;
#         polish_poses = []
#         polish_poses = self.generate_trajectory(pose1,pose2,100)
#         #pose3 = Pose()
#         #pose3 = deepcopy(pose1)
#         #pose3.position.z = pose1.position.z-0.2;
#         polish_poses.extend(self.generate_trajectory(pose2,pose1,100))
 
##########################################################################################
 
        # Command Current Joint Positions first
#         limb_interface = baxter_interface.limb.Limb(limb)
#         current_angles = [limb_interface.joint_angle(joint) for joint in limb_interface.joint_names()]
        current_angles = [self._limb.joint_angle(joint) for joint in self._limb.joint_names()]
        #current_angles = [self._arm.joint_angle(joint) for joint in self._arm.joint_names()]
        time = 0.0
        self.add_point(current_angles, time) # Set current configuration as first point in the trajectory
    
        current_pose = self._pose
        #print current_pose
        #print current_angles
      
#         #p1 = self.positions[limb]
#         #self.add_point(p1, 7.0)
#         #self.add_point([x * 0.75 for x in p1], 9.0)
#         #self.add_point([x * 1.25 for x in p1], 12.0)
#       
#         p0_angles = self.get_angles(self.get_pose_msg(current_pose), hover_distance=0.0)
#         #p0_list = {limb: p0_angles.values()}
#         p0 = p0_angles.values()
        p0_angles = self.get_angles_array(self.get_pose_msg(current_pose), hover_distance=0.2)
        duration = 5.0
        time = time + duration
        self.add_point(p0_angles, time)
        
        p1_angles = self.get_angles_array(self.get_pose_msg(current_pose), hover_distance=0.0)
        duration = 5.0
        time = time + duration
        self.add_point(p1_angles, time)
          
#         # Some Python functions and operations with lists and arrays
#         p1 = self.positions[limb]
#         #p1_matrix = np.array(p1)[np.newaxis] # Add a new dimension
#         #l1 = np.transpose(p1).tolist() # Transpose and convert to list
#         #list(p2) # Convert to list?
#         #a1 = np.asarray(p1) # List as array
#         duration = 5.0
#         time = time + duration
#         self.add_point(p1, time)
#      
#         #p2_angles = self.get_angles(self.get_pose_msg(current_pose), hover_distance=0.0)
#         p2_angles = self.get_angles(pose1, hover_distance=0.0)
#         p2 = p2_angles.values()  
        p2_angles = self.get_angles_array(pose02, hover_distance=0.0)
        duration = 5.0
        time = time + duration
        self.add_point(p2_angles, time)
          
        p3_angles = self.get_angles_array(pose02, hover_distance=-0.2)
        duration = 5.0
        time = time + duration
        self.add_point(p3_angles, time)
         
        p4_angles = self.get_angles_array(pose02, hover_distance=0.2)
        duration = 5.0
        time = time + duration
        self.add_point(p4_angles, time)
         
        p5_angles = self.get_angles_array(pose02, hover_distance=0.0)
        duration = 5.0
        time = time + duration
        self.add_point(p5_angles, time)
    
        # The Pose of the movement in its initial location.
        pose05 = deepcopy(pose02)
        pose05.position.x = pose05.position.x+0.1;
        pose06 = deepcopy(pose05)
        pose06.position.y = pose06.position.y-0.1;
        pose07 = deepcopy(pose06)
        pose07.position.x = pose07.position.x-0.1;
        pose08 = deepcopy(pose07)
        pose08.position.y = pose08.position.y+0.1;
        pts = 5
        poses = []
        poses_edge1 = self.generate_trajectory(pose02, pose05, pts)
        poses_edge2 = self.generate_trajectory(pose05, pose06, pts)
        poses_edge3 = self.generate_trajectory(pose06, pose07, pts)
        poses_edge4 = self.generate_trajectory(pose07, pose08, pts)
        poses.extend(poses_edge1)
        poses.extend(poses_edge2)
        poses.extend(poses_edge3)
        poses.extend(poses_edge4)
          
        point = []
        for i in range(len(poses)):
            #point.append(self.get_angles(pose[i], hover_distance=0.0))
            #self.add_point(point[i], 0.5)
            print i
            angles = self.get_angles_array(poses[i], hover_distance=0.0)
            duration = 2.0 # Duration per edge of the square
            time = time + duration/pts
            self.add_point(angles, time)
              
        #print self._goal.trajectory.points
              
        self.start()
        self.wait(time+5.0) # Extra time of 2 sec to complete the action
        print("Exiting - Joint Trajectory Action Test Complete")

#######################################################################


#     jacobian = traj._kin.jacobian()
#     jacobianInv = traj._kin.jacobian_pseudo_inverse()
#     u_cartesian = np.array([[0], [0], [0.05], [0], [0], [0]])
#     u_joints = jacobianInv * u_cartesian
#      
#     print '#########################'
#     #print traj._limb.endpoint_pose()
#     print traj._limb._joint_names['right']
#      
#     print traj._angles
#     print traj._pose
#     print '#########################'
#      
#     print '#########################'
#     print jacobian
#     print jacobianInv
#     print u_cartesian
#     print u_joints
#     print '#########################'
#  
#     #joint_velocities = traj._limb.joint_angles()
#     #print joint_velocities
#     joint_velocities = dict()
#      
#     joint_positions = traj._limb.joint_angles()


#     k0 = k1 = k2 = k3 = k4 = k5 = k6 = 0.01
#     kx = ky = kz = 0.1
#     tc = 0.01
#     print '#########################'
#     print traj._limb._joint_names[limb]
#     print '#########################'
#        
#     joint_velocities = traj._limb.joint_angles()
#     #print joint_velocities
#     #joint_velocities = dict()
#        
#     while not rospy.is_shutdown():
#            
#     #if True:
#         #print("\nPolishing...")
#         #traj.move_to_pose(polish_poses[idx],0.05)
#         #idx = (idx+1) % len(polish_poses)
#            
# #         current_pose = traj.get_pose_msg()
# #         current_joints = traj.ik_request(current_pose)
# #         ref_joints = traj.ik_request(polish_poses[idx])
# #         #print current_joints
# #         #print ref_joints
# #         
# #         e0=(ref_joints.values()[0]-current_joints.values()[0])/tc
# #         e1=(ref_joints.values()[1]-current_joints.values()[1])/tc
# #         e2=(ref_joints.values()[2]-current_joints.values()[2])/tc
# #         e3=(ref_joints.values()[3]-current_joints.values()[3])/tc
# #         e4=(ref_joints.values()[4]-current_joints.values()[4])/tc
# #         e5=(ref_joints.values()[5]-current_joints.values()[5])/tc
# #         e6=(ref_joints.values()[6]-current_joints.values()[6])/tc
# #         u0=k0*e0;
# #         u1=k1*e1;
# #         u2=k2*e2;
# #         u3=k3*e3;
# #         u4=k4*e4;
# #         u5=k5*e5;
# #         u6=k6*e6;
#    
#         current_pose_msg = traj.get_pose_msg(traj._pose)
#         #print current_pose_msg
#         ex = (polish_poses[idx].position.x-current_pose_msg.position.x)/tc
#         ey = (polish_poses[idx].position.y-current_pose_msg.position.y)/tc
#         ez = (polish_poses[idx].position.z-current_pose_msg.position.z)/tc
#         ux = kx*ex;
#         uy = ky*ey;
#         uz = kz*ez;
#            
#         jacobian = traj._kin.jacobian()
#         jacobianInv = traj._kin.jacobian_pseudo_inverse()
#         #jacobian = deepcopy(traj._kin.jacobian())
#         #jacobianInv = deepcopy(traj._kin.jacobian_pseudo_inverse())
#         #jacobianTra = jacobian.T
#         #jacobianTraInv = np.linalg.pinv(jacobianTra)
#         #u_cartesian = np.array([[ux], [uy], [uz], [0], [0], [0]])
#         u_cartesian = np.array([[0], [0], [0.05], [0], [0], [0]])
#         #print _kin.jacobian()
#         #print jacobianInv
#         #print u_cartesian
#    
#         u_joints = jacobianInv * u_cartesian
#         #u_joints = np.dot(jacobianInv, u_cartesian)
#         #u_joints = np.matmul(jacobianInv, u_cartesian)
#         #print u_joints
#            
# #        joint_velocities = dict(zip(traj._limb._joint_names['right'],
# #                          [u0, u1, u2, u3, u4, u5, u6]))
# #        joint_velocities = dict(zip(traj._limb._joint_names['right'],
# #                          u_joints))
# #        print joint_velocities
#         joint_velocities['right_s0'] = u_joints[0]
#         joint_velocities['right_s1'] = u_joints[1]
#         joint_velocities['right_e0'] = u_joints[2]
#         joint_velocities['right_e1'] = u_joints[3]
#         joint_velocities['right_w0'] = u_joints[4]
#         joint_velocities['right_w1'] = u_joints[5]
#         joint_velocities['right_w2'] = u_joints[6]
# #        joint_velocities['right_s0'] = u_joints.item(0)
# #        joint_velocities['right_s1'] = u_joints.item(1)
# #        joint_velocities['right_e0'] = u_joints.item(2)
# #        joint_velocities['right_e1'] = u_joints.item(3)
# #        joint_velocities['right_w0'] = u_joints.item(4)
# #        joint_velocities['right_w1'] = u_joints.item(5)
# #        joint_velocities['right_w2'] = u_joints.item(6)

#         print joint_velocities
#         traj.set_joint_velocity(joint_velocities)
#         rospy.sleep(tc)
#         #print idx

####################################################################################

#         jacobian = traj._kin.jacobian()
#         jacobianInv = traj._kin.jacobian_pseudo_inverse()
#  
#         u_cartesian = np.array([[0], [0], [0.2], [0], [0], [0]]) # vx, vy, vz, wx, wy, wz
#         u_joints = jacobianInv * u_cartesian
#         #u_joints = np.array([[0.05], [0], [0], [0], [0], [0], [0]]) # u0, u1, u2, u3, u4, u5, u6
#  
#         #joint_velocities = dict(zip(traj._limb._joint_names['right'],
#         #                  u_joints))
#  
# #         joint_velocities = dict(zip(traj._joint_names,
# #                           u_joints))
#  
# #        traj.set_joint_velocity(joint_velocities)
#          
# #        print joint_velocities
# #         joint_velocities['right_s0'] = u_joints[0]
# #         joint_velocities['right_s1'] = u_joints[1]
# #         joint_velocities['right_e0'] = u_joints[2]
# #         joint_velocities['right_e1'] = u_joints[3]
# #         joint_velocities['right_w0'] = u_joints[4]
# #         joint_velocities['right_w1'] = u_joints[5]
# #         joint_velocities['right_w2'] = u_joints[6]
#  
# #         #print joint_velocities
# #         traj.set_joint_velocity(joint_velocities)
# #         #traj._arm.set_joint_velocities(joint_velocities)
# #         #print idx
#           
#         ## POSITION CONTROL
#         traj._cmd.mode = traj._cmd.POSITION_MODE
#         #traj._cmd.names = traj._joint_names
#         traj._cmd.names = traj._angles.keys()
#         traj._cmd.command = traj._angles.values()
#          
#         #joint_positions = traj._angles.keys()
#         #joint_positions = traj._limb.joint_angles()
#         joint_positions = traj._limb.joint_angles()
#         traj.set_joint_position(joint_positions)
#  
#         #for i in range(0,7):
#         #    #traj._cmd.command[i] = traj._angles.value()[i]+u_joints[i]*tc
#         #    print str(i) + ' ' + str(u_joints.iloc[i].values) + ' ' #+ str(traj._angles.value()[i]) + ' ' + str(u_joints[i]*tc)
#         #for i in range(0,7):
#         #    traj._cmd.command[i] = 0#*traj._angles.values()
#          
#         #u_joints = np.array([[0.05], [0], [0], [0], [0], [0], [0]]) # u0, u1, u2, u3, u4, u5, u6
#          
# #         traj._cmd.command[0] = traj._cmd.command[0] + u_joints[0]*tc
# #         traj._cmd.command[1] = traj._cmd.command[1] + u_joints[1]*tc
# #         traj._cmd.command[2] = traj._cmd.command[2] + u_joints[2]*tc
# #         traj._cmd.command[3] = traj._cmd.command[3] + u_joints[3]*tc
# #         traj._cmd.command[4] = traj._cmd.command[4] + u_joints[4]*tc
# #         traj._cmd.command[5] = traj._cmd.command[5] + u_joints[5]*tc
# #         traj._cmd.command[6] = traj._cmd.command[6] + u_joints[6]*tc
#  
# #         traj._cmd.command[0] = traj._angles.values()[0] - 0
# #         traj._cmd.command[1] = traj._angles.values()[1] + 0
# #         traj._cmd.command[2] = traj._angles.values()[2] + 0
# #         traj._cmd.command[3] = traj._angles.values()[3] + 0
# #         traj._cmd.command[4] = traj._angles.values()[4] + 0
# #         traj._cmd.command[5] = traj._angles.values()[5] + 0
# #         traj._cmd.command[6] = traj._angles.values()[6] + 0
#  
#         #traj._cmd.command[0] = traj._cmd.command[0] + u_joints[0]*tc
#          
#          
# #         ## VELOCITY CONTROL
# #         traj._cmd.mode = traj._cmd.VELOCITY_MODE
# #         traj._cmd.names = traj._joint_names
# #         #traj._cmd.names = traj._angles.keys()
# #         #traj._cmd.command = [-0.2,0.2,0.2,0.2,0,0,0]
# #         traj._cmd.command = [0,0,0,0.2,0,0,0]
#          
#         ## PUBLISH CMD CONTROL
#         #traj._pub_cmd.publish(traj._cmd)
#          
# #         print traj._joint_names
# #         print traj._angles.keys()
# #         print traj._angles.values()
# #         print traj._cmd.command
#         #print '2' + traj._angles.keys()
#          
#         #rospy.sleep(tc)
#         rate.sleep()

####################################################################################

#     endpoint_pose = self._limb.endpoint_pose()
#     fk_pose=get_pose_msg(self, endpoint_pose)

####################################################################################

    def add_point(self, positions, time):
        point = JointTrajectoryPoint()
        point.positions = copy(positions)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)

    def stop(self):
        self._client.cancel_goal()

    def wait(self, timeout=15.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def result(self):
        return self._client.get_result()

    def clear(self, limb):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._goal.trajectory.joint_names = [limb + '_' + joint for joint in \
            ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]

    def _reset_control_modes(self):
        rate = rospy.Rate(self._rate)
        for _ in xrange(100):
            if rospy.is_shutdown():
                return False
            self._limb.exit_control_mode()
            #self._arm.exit_control_mode()
            self._pub_rate.publish(100)  # 100Hz default joint state rate
            rate.sleep()
        return True
    
    def clean_shutdown(self):
        """Handles ROS shutdown (Ctrl-C) safely."""
        print("\nExiting example...")
        if self._server:
            self.stop()
        #return to normal
        self._reset_control_modes()
        self.move_to_neutral()
        #if not self._init_state:
        #    print("Disabling robot...")
        #    self._rs.disable()
        return True

####################################################################################

def load_gazebo_models(table_pose=Pose(position=Point(x=1.0, y=0.0, z=0.0)),
                       table_reference_frame="world",
                       block_pose=Pose(position=Point(x=0.6725, y=0.1265, z=0.7825)),
                       block_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('baxter_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    # Load Block URDF
    block_xml = ''
    with open (model_path + "block/model.urdf", "r") as block_file:
        block_xml=block_file.read().replace('\n', '')
    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block", block_xml, "/",
                               block_pose, block_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("cafe_table")
        resp_delete = delete_model("block")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))

####################################################################################

def main():
    
####################################################################################
    
    """RSDK Joint Trajectory Example: Simple Action Client
   
    Creates a client of the Joint Trajectory Action Server
    to send commands of standard action type,
    control_msgs/FollowJointTrajectoryAction.
   
    Make sure to start the joint_trajectory_action_server.py
    first. Then run this example on a specified limb to
    command a short series of trajectory points for the arm
    to follow.
    """
#     """RSDK Joint Velocity Example: Cartesian control using Jacobian matrix
#   
#     Commands joint velocities to perform movements in the Cartesian space.
#     Demonstrates Joint Velocity Control Mode.
#     """
#     """RSDK Inverse Kinematics Pick and Place Example
#    
#     A Pick and Place example using the Rethink Inverse Kinematics
#     Service which returns the joint angles a requested Cartesian Pose.
#     This ROS Service client is used to request both pick and place
#     poses in the /base frame of the robot.
#    
#     Note: This is a highly scripted and tuned demo. The object location
#     is "known" and movement is done completely open loop. It is expected
#     behavior that Baxter will eventually mis-pick or drop the block. You
#     can improve on this demo by adding perception and feedback to close
#     the loop.
#     """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-l', '--limb', required=True, choices=['left', 'right'],
        help='send joint trajectory to which limb'
    )
    args = parser.parse_args(rospy.myargv()[1:])
    limb = args.limb
     
    print("Initializing node... ")
    rospy.init_node("rsdk_joint_trajectory_client_%s" % (limb,))
     
    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    ##load_gazebo_models()
    # Remove models from the scene on shutdown
    ##rospy.on_shutdown(delete_gazebo_models)
   
    #traj = TrajectoryControl(limb, sim = True)
    traj = TrajectoryControl(limb, sim = False, server = False)
    rospy.on_shutdown(traj.clean_shutdown)
    #rospy.on_shutdown(traj.stop)

##########################################################################################
     
    # Move the Baxter to the untuck position
#     traj._untuck.supervised_tuck()
#     rospy.sleep(1.0)

#     # Move to neutral angles
#     traj.move_to_neutral(timeout=3.0)

#     # Pick and place demo
#     traj.pick_place()
    
#     # Wooble arms
#     traj.wobble()
      
    # Control loop to test different controllers  
    traj.loop()

#     traj.gripper_calibrate()
#     rospy.sleep(1.0)

#     # Trajectory action
#     traj.action()
    
    
####################################################################################
    
    return 0

####################################################################################

if __name__ == "__main__":
    main()

#      rospy.init_node("ik_pick_and_place_demo")
#      limb='right'
#      _kin = baxter_kinematics(limb)
#      jacobian = _kin.jacobian()
#      print jacobian
#      limb_interface = baxter_interface.limb.Limb(limb)
#      current_angles = [limb_interface.joint_angle(joint) for joint in limb_interface.joint_names()]
#      print current_angles
#      print limb_interface.joint_angle['right']
#      print limb_interface.joint_names[1:-1]

#     from urdf_parser_py.urdf import URDF
#     from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
#     from pykdl_utils.kdl_kinematics import KDLKinematics
# 
#     robot = URDF.load_from_parameter_server(verbose=False)
#     tree = kdl_tree_from_urdf_model(robot)
#     print tree.getNrOfSegments()
#     chain = tree.getChain(base_link, end_link)
#     print chain.getNrOfJoints()
#     
#     robot = URDF.from_parameter_server()
#     kdl_kin = KDLKinematics(robot, base_link, end_link)
#     q = kdl_kin.random_joint_angles()
#     pose = kdl_kin.forward(q) # forward kinematics (returns homogeneous 4x4 numpy.mat)
#     q_ik = kdl_kin.inverse(pose, q+0.3) # inverse kinematics
#     if q_ik is not None:
#         pose_sol = kdl_kin.forward(q_ik) # should equal pose
#     J = kdl_kin.jacobian(q)
#     print 'q:', q
#     print 'q_ik:', q_ik
#     print 'pose:', pose
#     if q_ik is not None:
#         print 'pose_sol:', pose_sol
#     print 'J:', J
