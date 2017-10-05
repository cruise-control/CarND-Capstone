#!/usr/bin/env python

import rospy

from geometry_msgs.msg import PoseStamped, Point

from visualization_msgs.msg import Marker

# See following tutorial for reference
# http://library.isr.ist.utl.pt/docs/roswiki/rviz(2f)Tutorials(2f)Markers(3a20)Basic(20)Shapes.html
class Visualization(object):

    def __init__(self):
        rospy.init_node('visualization')
        #rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        #rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        self.visulization_pub = rospy.Publisher('/pose_vis2', Marker, queue_size=1)

        
        while not rospy.is_shutdown():
            self.pose_cb()
            print self.visulization_pub.get_num_connections()

        rospy.spin()

    def pose_cb(self):
        marker = Marker()
        marker.header.frame_id = "/world"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = marker.CUBE
        marker.scale.x = 5
        marker.scale.y = 5
        marker.scale.z = 5
        marker.ns = "car"

        marker.action = marker.ADD
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = 1.
        marker.pose.position.y = 1.
        marker.pose.position.z = 0.

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.b = 1.0
        marker.lifetime = rospy.Duration(0)

        #for i in range(10):
        #    p = Point()
        #    p.x = i
        #    p.y = 0.5
        #    p.z = 0.5
        #    marker.points.append(p)
        #marker.pose = msg.pose
        self.visulization_pub.publish(marker)
        #x = msg.pose.position.x
        #y = msg.pose.position.y
        #z = msg.pose.position.z
        rospy.sleep(1)



if __name__ == '__main__':
    try:
        Visualization()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start visualization node.')