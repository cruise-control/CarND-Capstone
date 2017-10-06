#!/usr/bin/env python

import rospy
import tf

from geometry_msgs.msg import PoseStamped, Point
from styx_msgs.msg import Lane, Waypoint
from visualization_msgs.msg import Marker

# See following tutorial for reference
# http://library.isr.ist.utl.pt/docs/roswiki/rviz(2f)Tutorials(2f)Markers(3a20)Basic(20)Shapes.html
class Visualization(object):

    def __init__(self):
        rospy.init_node('visualization')
        #rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/final_waypoints', Lane, self.waypoints_cb)

        self.visualization_pub = rospy.Publisher('/pose_vis2', Marker, queue_size=1)
        #self.transform = tf.TransformListener()

        #self.pose = PoseStamped()
        #while not rospy.is_shutdown():
        #    self.pose_cb()
            #print self.visualization_pub.get_num_connections()
        #self.counter = 0
        rospy.spin()


    def waypoints_cb(self, msg):
        if self.visualization_pub.get_num_connections() > 0:
            
            # Marker is a visualization msg for rviz
            marker = Marker()
            
            # Base link refers to vehicle coordinate system
            marker.header.frame_id = '/world'
            marker.header.stamp = rospy.Time.now()
            marker.id = 0
            marker.type = marker.SPHERE_LIST
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.ns = "car"

            marker.action = marker.ADD
            #marker.pose.orientation.x = 0.
            #marker.pose.orientation.y = 0.
            #marker.pose.orientation.z = 0.
            #marker.pose.orientation.w = 1.0
            #marker.pose.position.x = 0. #msg.pose.position.x
            #marker.pose.position.y = 0. #msg.pose.position.y
            #marker.pose.position.z = 0.
            
            # Set RGB-A channels
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Zero lifetime means it won't be deleted
            marker.lifetime = rospy.Duration(1)
            
            #print 'msg frame_id', msg.waypoints[0].pose.header.frame_id
            for waypoint in msg.waypoints:
            #    waypoint.pose.header.frame_id = '/base_link'
            #    #try:
            #     = self.transform.transformPose('/world', waypoint.pose)
            #    #except tf.LookupException: 
                #    print 'WTF'
                p = Point()
                p = waypoint.pose.pose.position
                marker.points.append(p)
            self.visualization_pub.publish(marker)
            rospy.sleep(2)



    def pose_cb(self, msg):
        self.pose = msg.pose
        rospy.sleep(2)



if __name__ == '__main__':
    try:
        Visualization()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start visualization node.')