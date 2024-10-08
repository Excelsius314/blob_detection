import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int16
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
class BlobDetector(Node):

    def __init__(self):
        super().__init__('cv2_blob_detector')
        self.bridge = CvBridge()

        # "red (0,15)"
        # yellow: 25, 30)
        self.color_angles = {"red" : (0, 15), "yellow": (25, 30), "blue": (100, 150)}
        
        self.declare_parameter('robot_names', value=["epuck2"])
        self.robot_names = self.get_parameter("robot_names").get_parameter_value().string_array_value
        self.get_logger().info('Robot_names: {}'.format(self.robot_names))

        self.declare_parameter('corresponding_colors', value=["yellow"])
        self.colors = self.get_parameter('corresponding_colors').get_parameter_value().string_array_value
        self.color_to_robot_name = {color:robot for color, robot in zip(self.colors, self.robot_names)}

        self.get_logger().info("Robot - earch color association: {}".format(self.color_to_robot_name))


        self.detection_publishers = {color: self.create_publisher(Image, "{}/blob_detections".format(self.color_to_robot_name[color]), 10) for color in self.colors}

        def get_detec_blob_cb(color):

            def detect_blob(msg):
                return self.detect_blobs(msg, color)
            
            return detect_blob

        self.color_debuggers = {color: self.create_publisher(Image, "/color_debug/{}".format(color), 10) for color in self.colors}
        self.image_subs = [self.create_subscription(CompressedImage, "/{}/image_raw/compressed".format(self.color_to_robot_name[color]),
                                                    get_detec_blob_cb(color), 1) for color in self.colors]
        #self.image_subs = [self.create_subscription(CompressedImage, "/image_raw/compressed",
        #                                            get_detec_blob_cb(color), 1) for color in self.colors]
        
        self.get_logger().info("Subs: {}".format([sub.topic_name for sub in self.image_subs]))
        self.color_center_pubs = {color: self.create_publisher(Int16, "/{}/ball_coordinates".format(self.color_to_robot_name[color]), 10) for color in self.colors}

        self.draw_colors = {"red": (255, 0, 0), "blue": (0, 0, 255), "yellow": (0, 255, 255)}
        self.prev_known_centers = {color: None for color in self.colors}


        self.blob_count_treshhold = 10
        self.blob_bin_size = 10
        self.min_saturation = 120
        self.border_point_bin_size = 10

    
    def get_biggest_blob_via_hist(self, coords, max_val):


        counts, bins = np.histogram(coords, bins=int(max_val/self.blob_bin_size))
        largest_bin_range = [0, 0]
        largest_bin_count = 0

        current_bin_range = [0, 0]
        current_bin_count = 0
        current_bin_active = False


        all_bins = []

        for i in range(len(counts)):
            count = counts[i]

            # Continue or start bin
            if count >= self.blob_count_treshhold and not (i ==len(counts) -1):
                if not current_bin_active:
                    current_bin_range[0] = i
                    current_bin_active = True

                current_bin_range[1] = i
                current_bin_count += count
            else:
                if count >= self.blob_count_treshhold:
                    current_bin_range[1] = i
                    current_bin_count += count  
        
                 # end bin
                if current_bin_count > largest_bin_count:
                    largest_bin_range = current_bin_range.copy()
                    largest_bin_count = current_bin_count

                    all_bins.append((current_bin_range.copy(), current_bin_count))
                current_bin_active = False
                current_bin_count = 0
        
        return bins[largest_bin_range[0]], bins[largest_bin_range[1]+1]


    def calculate_blob_center(self, valid_points, h, w):
            

            #single_color_img[non_ball] = 0


            largest_bin_height = self.get_biggest_blob_via_hist(valid_points[0], h)
            largest_bin_width = self.get_biggest_blob_via_hist(valid_points[1], w)
            in_largest_blob_height = np.logical_and(valid_points[0] > largest_bin_height[0], valid_points[0] < largest_bin_height[1])
            in_largest_blob_width = np.logical_and(valid_points[1] > largest_bin_width[0], valid_points[1] < largest_bin_width[1])
            in_largest_blob = np.logical_and(in_largest_blob_height,in_largest_blob_width)
    
            valid_points_h = valid_points[0][in_largest_blob]
            valid_points_w = valid_points[1][in_largest_blob]

            combined_valids = np.vstack((valid_points_h, valid_points_w)).transpose().astype(float)
            print(combined_valids)
            if len(combined_valids) == 0:
                return None

                fig = plt.figure()
                plt.scatter(valid_points[1], h-valid_points[0])
                plt.scatter(
                    [np.average(valid_points[1])],
                    [h - np.average(valid_points[0])],
                    marker="o",
                    c="red",
                    alpha=1,
                    s=200,
                    edgecolor="k",
                )
                plt.xlim((0, w))
                plt.ylim((0, h))

                fig = plt.figure()
                plt.scatter(valid_points[1][in_largest_blob], h-valid_points[0][in_largest_blob])

                fig = plt.figure()
                counts, bins = np.histogram(valid_points[0], bins=int(h/10))
                plt.stairs(counts, bins)
                plt.title("Y-Histogramm")

                fig = plt.figure()
                counts, bins = np.histogram(valid_points[1], bins=int(w/10))
                plt.stairs(counts, bins)
                plt.title("X-Histogramm")

                fig = plt.figure()
                plt.title("Only largest blob points")
                #plt.scatter(border_points[:, 1], h-border_points[:, 0], color="red")
                #print("Results: {}".format(res.x))
                #plt.scatter([res.x[1]], [h - res.x[0]], color="blue")
                #plt.scatter([m_w], [h - m_h], color="green")

                plt.show()
                return

            border_points = []

            min_height, max_height = np.min(valid_points_h), np.max(valid_points_h)
            height_bins = np.linspace(min_height, max_height, int((max_height - min_height)/self.border_point_bin_size))
            for i in range(len(height_bins) -1):

                points_on_height_level = combined_valids[np.logical_and(combined_valids[:, 0] < height_bins[i+1],
                                                                        combined_valids[:, 0] >= height_bins[i])]
                if len(points_on_height_level) == 0:
                    continue

                height = (height_bins[i+1] + height_bins[i])/2
                border_points.append([height, np.min(points_on_height_level[:, 1])])
                border_points.append([height, np.max(points_on_height_level[:, 1])])
            
            border_points = np.array(border_points)

            if len(border_points) == 0:
                return None

            # m_h, m_w, rad
            def f(vars):
                return np.sum(((border_points[:, 0] - vars[0]) ** 2 + (border_points[:, 1] - vars[1]) ** 2 - vars[2]**2) **2)

            # estimate intial values
            m_h, m_w = np.average(border_points[:, 0]), np.average(border_points[:, 1])
            r = np.average(np.sqrt((border_points[:, 0] - m_h) ** 2 +  (border_points[:, 1] - m_w) ** 2))
            res = least_squares(f, np.array([m_h.copy(), m_w.copy(), r.copy()]))


            return np.array(res.x[:2])




    def detect_blobs(self, img_msg, color):
        
        #cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

        try:
             if type(img_msg) is CompressedImage:
                 #np_arr = np.fromstring(img_msg.data, np.uint8)
                 #cv_image = cv.imdecode(np_arr, cv.IMREAD_COLOR)  # decode
                 cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
             else:
                 cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")

        except CvBridgeError as e:
            self.get_logger.error(e)


        #self.get_logger().info("Received img from {}".format(color))
        h,w = cv_image.shape[0], cv_image.shape[1]

        #color_corrected = cv_image.copy().astype(float)
        #intensities = np.sum(color_corrected, axis=2)
        #color_corrected[:, :, 1] = color_corrected[:, :, 1] * 0.75
        #new_intensities = np.sum(color_corrected, axis=2)
        #color_corrected = np.clip(color_corrected * (intensities/new_intensities).reshape(h, w, 1), 0, 255)
        #color_corrected = np.floor(color_corrected)
        #cv_image = color_corrected.astype("uint8")

        # hsv
        hsv_im = cv.cvtColor(cv_image, cv.COLOR_RGB2HSV)
        is_not_a_color = hsv_im[:, : , 1] < self.min_saturation

        color_range = self.color_angles[color]
        is_not_correct_color = np.logical_or(hsv_im[:, : , 0] > color_range[1], hsv_im[:, : , 0] < color_range[0])
        non_ball = np.logical_or(is_not_a_color, is_not_correct_color)

        single_color_img =  hsv_im.copy()
        single_color_img[non_ball] *= 0
        debug_msg = self.bridge.cv2_to_imgmsg(cv.cvtColor(single_color_img, cv.COLOR_HSV2RGB), encoding='rgb8')
        debug_msg.header = img_msg.header
        self.color_debuggers[color].publish(debug_msg)

        valid_points = np.logical_not(non_ball).nonzero()
        center = self.calculate_blob_center(valid_points, h, w)

        if center is None:
            if self.prev_known_centers[color] is not None:
                center = self.prev_known_centers[color]
            else:
                center = np.array([0, 0])
        else:
            #if self.prev_known_centers[color] is not None:
            #    center = center/4 + 3*self.prev_known_centers[color]/4
            self.prev_known_centers[color] = center

            cv_image = cv.circle(cv_image, (int(center[1]), int(center[0])), 2, self.draw_colors[color], 1)

            coord_msg = Int16()
            coord_msg.data = int(w/2 - center[1])
            self.color_center_pubs[color].publish(coord_msg)

        print("Publish results")
        result_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')
        result_msg.header = img_msg.header
        self.detection_publishers[color].publish(result_msg)


def main(args=None):

    rclpy.init(args=args)
    blob_detector = BlobDetector()
    rclpy.spin(blob_detector)
    blob_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
