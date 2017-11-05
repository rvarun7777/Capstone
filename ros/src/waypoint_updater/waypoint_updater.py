#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight, TrafficLightArray
from std_msgs.msg import Int32, Bool
import math
import copy
import tf

LOOKAHEAD_WPS = 100  # Number of waypoints we will publish. You can change this number
POSITION_SEARCH_RANGE = 10  # Number of waypoints to search current position back and forth

MAX_DECEL = 1.
MAX_ACCEL = 1.

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb, queue_size=1)
        
        self.dbw_enabled = True

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.next_wp_pub = rospy.Publisher('/next_wp', Int32, queue_size=1)

        # current path
        # NOTE: supposedly comes from top level planner (from file for simulator) at rate 40Hz
        self.waypoints = None

        # current pose
        # NOTE: supposedly comes from fusion (briged from simulator) at unknown rate
        self.current_pose = None
        self.next_waypoint_index = None

        self.stop_trajectory = None
        self.start_trajectory = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if (self.waypoints and self.next_waypoint_index):
                lane = Lane()
                lane.header.frame_id = self.current_pose.header.frame_id
                lane.header.stamp = rospy.Time(0)
                lane.waypoints = self.waypoints[self.next_waypoint_index:self.next_waypoint_index+LOOKAHEAD_WPS]

                if (self.stop_trajectory):
                    start_index = self.stop_trajectory[0]
                    wps = self.stop_trajectory[1]
                    shift = self.next_waypoint_index - start_index
                    for i in range(min(LOOKAHEAD_WPS, len(lane.waypoints))):
                        shifted_i = i + shift
                        lane.waypoints[i] = wps[shifted_i] if (0 <= shifted_i < len(wps)) else lane.waypoints[i]
                elif self.start_trajectory and len(self.start_trajectory[1])>0:
                    start_index = self.start_trajectory[0]
                    wps = self.start_trajectory[1]
                    shift = self.next_waypoint_index - start_index
                    for i in range(min(LOOKAHEAD_WPS, len(lane.waypoints))):
                        shifted_i = i + shift
                        lane.waypoints[i] = wps[shifted_i] if (0 <= shifted_i < len(wps)) else lane.waypoints[i]

                self.final_waypoints_pub.publish(lane)
            rate.sleep()

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg.data
        if not self.dbw_enabled:
            self.stop_trajectory = None

    def euclidean_distance_2d(self, position1, position2):
        a = position1
        b = position2
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    def get_closest_waypoint(self):
        min_dist = 100000
        min_ind = 0
        ind = 0

        start = 0
        end = len(self.waypoints)
        if (self.next_waypoint_index):
            start = max (self.next_waypoint_index - POSITION_SEARCH_RANGE, 0)
            end = min (self.next_waypoint_index + POSITION_SEARCH_RANGE, end)

        position1 = self.current_pose.pose.position
        for i in range(start, end):
            position2 = self.waypoints[i].pose.pose.position
            dist = self.euclidean_distance_2d(position1, position2)
            if dist < min_dist:
                min_dist = dist
                min_ind = i
        return min_ind

    def current_yaw(self):
        quaternion = (
            self.current_pose.pose.orientation.x,
            self.current_pose.pose.orientation.y,
            self.current_pose.pose.orientation.z,
            self.current_pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        return euler[2]

    def update_next_waypoint(self):
        ind = self.get_closest_waypoint()

        map_x = self.waypoints[ind].pose.pose.position.x
        map_y = self.waypoints[ind].pose.pose.position.y

        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y

        yaw = self.current_yaw()

        x_car_system = ((map_x-x) * math.cos(yaw) + (map_y-y) * math.sin(yaw))
        if ( x_car_system < 0. ):
            ind += 1
        self.next_waypoint_index = ind
        return ind

    def pose_cb(self, msg):
        self.current_pose = msg
        if self.waypoints:
            next_waypoint_index = self.update_next_waypoint()
            self.next_wp_pub.publish(Int32(next_waypoint_index))


    def waypoints_cb(self, lane):
        if hasattr(self, 'waypoints') and self.waypoints != lane.waypoints:
            self.waypoints = lane.waypoints
            # full search current position on each update of waypoints
            self.next_waypoint_index = None

    def set_stop_trajectory(self, next_waypoint_index, stop_line_index):
        if (self.stop_trajectory):
            old_start = self.stop_trajectory[0]
            wps = self.stop_trajectory[1]
            self.stop_trajectory = [next_waypoint_index, wps[next_waypoint_index-old_start:]]
        else:
            if (stop_line_index >= next_waypoint_index):
                distance = self.euclidean_distance_2d(self.current_pose.pose.position, self.waypoints[next_waypoint_index].pose.pose.position)
                stop_distance = distance + self.distance(self.waypoints, next_waypoint_index, stop_line_index)
                full_stop_velocity = math.sqrt(2 * MAX_DECEL * stop_distance)
                target_velocity = self.waypoints[next_waypoint_index].twist.twist.linear.x
                v0 = min(full_stop_velocity, target_velocity)
                wps = []
                final_index = min(next_waypoint_index+2*LOOKAHEAD_WPS, len(self.waypoints))
                for i in range(next_waypoint_index, final_index):
                    velocity_setpoint = v0 * (max(stop_distance-distance, 0.))/stop_distance
                    if velocity_setpoint < 0.1:
                        velocity_setpoint = 0.
                    wp = copy.deepcopy(self.waypoints[i])
                    wp.twist.twist.linear.x  = velocity_setpoint
                    wps.append(wp)
                    if (i < stop_line_index):
                        distance += self.distance(self.waypoints, i, i + 1)

                self.stop_trajectory = [next_waypoint_index, wps]

    def set_start_trajectory(self, next_waypoint_index):
        if (self.start_trajectory):
            old_start = self.start_trajectory[0]
            wps = self.start_trajectory[1]
            self.start_trajectory = [next_waypoint_index, wps[next_waypoint_index-old_start:]]
        else:
            target_velocity = self.waypoints[next_waypoint_index].twist.twist.linear.x
            current_velocity = 0.
            if self.stop_trajectory:
                old_start = self.stop_trajectory[0]
                wps = self.stop_trajectory[1]
                cur_index = next_waypoint_index-old_start
                if len(wps)>cur_index:
                    current_velocity = wps[cur_index].twist.twist.linear.x
            distance = self.euclidean_distance_2d(self.current_pose.pose.position, self.waypoints[next_waypoint_index].pose.pose.position)
            wps = []
            for i in range(next_waypoint_index, len(self.waypoints)):
                velocity_setpoint = math.sqrt(2 * distance * MAX_ACCEL + current_velocity*current_velocity) 
                if i > next_waypoint_index+LOOKAHEAD_WPS or velocity_setpoint>target_velocity:
                    break
                wp = copy.deepcopy(self.waypoints[i])
                wp.twist.twist.linear.x  = velocity_setpoint
                wps.append(wp)
                current_velocity = velocity_setpoint
                if i+1 < len(self.waypoints):
                    distance = self.euclidean_distance_2d(self.waypoints[i].pose.pose.position, self.waypoints[i+1].pose.pose.position)

            self.start_trajectory = [next_waypoint_index, wps]
        self.stop_trajectory = None

    def get_vels(self, traj):
        res = None
        if traj:
            start = traj[0]
            vels = []
            for wp in traj[1]:
                vels.append(wp.twist.twist.linear.x)
            return [start, vels]
        return res

    def traffic_cb(self, traffic_waypoint):
        if (self.next_waypoint_index is None):
            return
        if (self.waypoints is None):
            return

        stop_line_index = traffic_waypoint.data
        if (stop_line_index > -1):
            self.start_trajectory = None
            self.set_stop_trajectory(self.next_waypoint_index, stop_line_index)
            #rospy.logerr("stop_trajectory: %s\n ", self.get_vels(self.stop_trajectory))
        else:
            #self.stop_trajectory = None
            self.set_start_trajectory(self.next_waypoint_index)
            #rospy.logerr("start_trajectory: %s\n ", self.get_vels(self.start_trajectory))


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def distance(self, waypoints, wp1, wp2):
        dist = 0

        def dl(a, b): return math.sqrt((a.x - b.x) **
                                       2 + (a.y - b.y)**2 + (a.z - b.z)**2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position,
                       waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
