#!/usr/bin/env python

import rospy

from interactive_markers.interactive_marker_server import *
from geometry_msgs.msg import PoseStamped

from visualization_msgs.msg import Marker


class Visualization(object):

    def __init__(self):
        rospy.init_node('visualization')
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        #rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        self.visulization_pub = rospy.Publisher('pose_vis', Marker, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        marker = Marker()
        marker.header.frame_id = "/world"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = marker.POINTS
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.action = marker.ADD

        marker.pose = msg.pose
        self.visulization_pub.publish(marker)
        #x = msg.pose.position.x
        #y = msg.pose.position.y
        #z = msg.pose.position.z



if __name__ == '__main__':
    try:
        Visualization()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start visualization node.')