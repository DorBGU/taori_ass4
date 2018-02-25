#!/usr/bin/env python
import numpy as np
import ctypes
import struct
import rospy
import roslib
import time
import math
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan, Image, JointState, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import openni2_camera
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs import *


rospy.init_node('robot_whisperer', anonymous=True)
velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
OBSTACLE_DISTANCE_LIMIT = 0.5
start_travel_center_dist = -1
current_center_dist = -1
drive_flag = False
stable_flag = True
distance_to_red_object_flag = True
distance = float('inf')
here = True
depth_bool = True
turn_to_door_flag = False
turn_to_door_angle = float('inf')
finish = False




x = -1
y = -1

def depth_callback(data):
    global distance, depth_bool
    depth_bool = False
    print "im in the depth"
    bridge = CvBridge()
    d_im = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    print d_im

    if x > -1:
        print d_im[y, x]
        distance = d_im[y, x]



def analyze_image(image):
    global x, y, depth_bool
    # print "start func"

    bridge = CvBridge()
    im = bridge.imgmsg_to_cv2(image, "bgr8")

    # red color boundaries (R,B and G)
    lower = [1, 0, 60]
    upper = [60, 40, 255]

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # change image to hsv and get the mask by lower and upper
    mask = cv2.inRange(im, lower, upper)
    output = cv2.bitwise_and(im, im, mask=mask)

    # cv2.imshow("Result", mask)
    # cv2.waitKey(33)

    et, thresh = cv2.threshold(mask, 40, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that we found
        cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)
        ((cen_x, cen_y), rad) = cv2.minEnclosingCircle(c)

        if rad > 5:

            cv2.circle(im, (int(cen_x), int(cen_y)), int(rad), (0, 255, 0), 2)
            x = int(cen_x)
            y = int(cen_y)

            depth_bool = True
            sub = rospy.Subscriber("/torso_camera/depth_registered/image_raw", Image, depth_callback)
            # print "subs to depth"
            while depth_bool:
                continue

            sub.unregister()


def red_object_callback(data):
    global distance_to_red_object_flag, distance, here

    if not here:
        return

    here = False
    analyze_image(data)
    distance_to_red_object_flag = False


def move_forward_callback(data):

    global drive_flag, current_center_dist, OBSTACLE_DISTANCE_LIMIT, start_travel_center_dist
    center = data.ranges[len(data.ranges) / 2]  # Maybe scan some k items and see if one or more of them is too close?

    if start_travel_center_dist < 0:  # Init robot distance from nearest object
        start_travel_center_dist = center

    rospy.loginfo(center)
    msg = Twist()
    diff = abs(center - start_travel_center_dist)

    if center > OBSTACLE_DISTANCE_LIMIT and diff < 0.5 and drive_flag:
        msg.linear.x = 0.5 - diff
    else:
        msg.linear.x = 0.0
        drive_flag = False
    velocity_pub.publish(msg)
    time.sleep(0.001)

'''
def turn_callback(data, angle):
    print "in da callback, angle: " + str(angle)

    global stable_flag
    msg = Twist()

    if(angle == 0):
        stable_flag = False
        return

    # Converting from angles to radians
    angular_speed = (abs(angle)/angle)*10 * 2 * math.pi / 360
    relative_angle = abs(angle) * 2 * math.pi / 360


    msg.angular.z = angular_speed

    if not stable_flag:
        # Setting the current time for distance calculus
        current_angle = 0
        t0 = rospy.Time.now().to_sec()

        while current_angle < relative_angle:
            velocity_pub.publish(msg)

            current_angle = abs(angular_speed) * (rospy.Time.now().to_sec() - t0)
            # msg.angular.z = msg.angular.z*0.9
            # t1 = rospy.Time.now().to_sec()

        msg.angular.z = 0
        velocity_pub.publish(msg)
        stable_flag = True


'''


def move_forward():
    global start_travel_center_dist, drive_flag
    start_travel_center_dist = -1
    drive_flag = True
    sub = rospy.Subscriber("/scan", LaserScan, move_forward_callback)

    while drive_flag:
        continue

    sub.unregister()



def turn(angle):
    print "im here nigga" + str(angle)
    global start_travel_center_dist, stable_flag
    start_travel_center_dist = current_center_dist
    stable_flag = False

    print "in da callback, angle: " + str(angle)

    
    msg = Twist()

    if (angle == 0):
        stable_flag = False
        return

    # Converting from angles to radians
    angular_speed = (abs(angle) / angle) * 10 * 2 * math.pi / 360
    relative_angle = abs(angle) * 2 * math.pi / 360

    msg.angular.z = angular_speed

    if not stable_flag:
        # Setting the current time for distance calculus
        current_angle = 0
        t0 = rospy.Time.now().to_sec()

        while current_angle < relative_angle:
            velocity_pub.publish(msg)

            current_angle = abs(angular_speed) * (rospy.Time.now().to_sec() - t0)
            # msg.angular.z = msg.angular.z*0.9
            # t1 = rospy.Time.now().to_sec()

        msg.angular.z = 0
        velocity_pub.publish(msg)
        stable_flag = True


def distance_to_red_object():
    global distance_to_red_object_flag, distance, here, x, y
    x = y = -1
    here = True
    distance_to_red_object_flag = True
    distance = float('inf')

    sub = rospy.Subscriber("/torso_camera/rgb/image_raw", Image, red_object_callback)

    while distance_to_red_object_flag:
        pass

    sub.unregister()

    if distance < float('inf'):
        return distance
    else:
        return -1


def find_red_object():
    degrees = 60
    counter = 1
    while distance_to_red_object() == -1 and degrees*counter < 360+40:
        turn(degrees)
        counter = counter+1
    if distance_to_red_object() < float('inf'):
        return distance_to_red_object()


def turn_to_door_callback(data):
    global turn_to_door_flag, turn_to_door_angle

    if turn_to_door_flag:
        turn_to_door_flag = False

        i = -1
        j = -1
        divider = 30
        step = len(data.ranges)/divider
        safe_distance = 1.5

        for k in range(divider - 1):
            if abs(data.ranges[int(k*step)] - data.ranges[int((k+1)*step)]) > safe_distance:
                if i == -1:
                    i = k
                else:
                    j = k

                    break

        spect = (abs(data.angle_min) + abs(data.angle_max)) * 180 / math.pi
        print str(i) + "is the left bound"
        print str(j) + "is the right bound"
        angle = ((i + j)/2) * spect/divider - spect/2
        print "angle is " + str(angle)
        turn_to_door_angle = angle
        turn_to_door_flag = False

        turn(angle)




def turn_to_door():
    global turn_to_door_flag, turn_to_door_angle
    turn_to_door_flag = True
    turn_to_door_angle = float('inf')


    sub = rospy.Subscriber("/scan", LaserScan, turn_to_door_callback)

    while turn_to_door_flag:
        continue
    sub.unregister()






def mission_tears_in_heaven():
    goal = MoveBaseGoal()
    move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")

    # Wait 60 seconds for the action server to become available
    move_base.wait_for_server(rospy.Duration(60))

    # Use the map frame to define goal poses
    goal.target_pose.header.frame_id = 'map'

    # Set the time stamp to "now"
    goal.target_pose.header.stamp = rospy.Time.now()

    # Set the goal pose to the i-th waypoint
    mypose = Pose(Point(12.003, -0.826, 0.000), Quaternion(0.000, 0.000, 0.541, 0.841))

    # Start the robot moving toward the goal
    goal.target_pose.pose = mypose

    print "goal sent, begin movement"
    move_base.send_goal(goal)

    # Allow 5 minutes to get there
    finished_within_time = move_base.wait_for_result(rospy.Duration(300))

    # Check for success or failure
    if not finished_within_time:
        move_base.cancel_goal()
        rospy.loginfo("Timed out achieving goal")
    else:
        print ""
        state = move_base.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal succeeded!")

            rospy.loginfo("State:" + str(state))
            print find_red_object()
        else:
            rospy.loginfo("Goal failed with error code:"+ str(state))


def move(goal):
    # Send the goal pose to the MoveBaseAction server
    self.move_base.send_goal(goal)

    # Allow 1 minute to get there
    finished_within_time = self.move_base.wait_for_result(rospy.Duration(60))

    # If we don't get there in time, abort the goal
    if not finished_within_time:
        self.move_base.cancel_goal()
        rospy.loginfo("Timed out achieving goal")
    else:
        # We made it!
        state = move_base.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal succeeded!")

def ui():
    user_input_str = ''

    while user_input_str != "q":
        user_input_str = raw_input(
            "enter command:\n1. mission_tears_in_heaven\n2. turn\n3. distance to red object\n4 find red object\n5. turn_to_door\nq. quit\n")

        if user_input_str == "1":
            mission_tears_in_heaven()
        elif user_input_str == "2":
            angle = float(raw_input('enter rotation angle'))
            turn(angle)
        elif user_input_str == "3":
            dist = distance_to_red_object()
            if dist == -1:
                print 'NULL'
            else:
                print dist
        elif user_input_str == "4":
            print find_red_object()
        elif user_input_str == "5":
            print turn_to_door()





ui()

