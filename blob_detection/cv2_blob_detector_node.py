import cv2 as cv
from cv_bridge import CvBridge
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int16
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

class BlobDetector(Node):

    def __init__(self):
        super().__init__('cv2_blob_detector')
        self.img_subscriber = self.create_subscription(Image, "/image_raw", self.detect_blobs, 1)
        self.bridge = CvBridge()

        self.detection_publisher = self.create_publisher(Image, "/blob_detections", 10)
        


        self.color_angles = {"red" : (0, 15), "green": (65, 80), "blue": (100, 150)}
        self.colors = ["red", "blue", "green"]

        self.prev_known_centers = {color: (0,0) for color in self.colors}
        self.color_debuggers = {color: self.create_publisher(Image, "/color_debug/{}".format(color), 10) for color in self.colors}
        self.color_center_pubs = {color: self.create_publisher(Int16, "/x_position/{}".format(color), 10) for color in self.colors}

        self.blob_count_treshhold =10
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


            return res.x[:2]




    def detect_blobs(self, img_msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

        h,w = cv_image.shape[0], cv_image.shape[1]
        #cv_image[:,:, 1] = (0.95* cv_image[:,:, 1]).astype("uint8")

        # hsv
        hsv_im = cv.cvtColor(cv_image, cv.COLOR_RGB2HSV)
        is_not_a_color = hsv_im[:, : , 1] < self.min_saturation

        print(hsv_im[is_not_a_color])

        centers = []

        for color in self.colors:

            color_range = self.color_angles[color]

            is_not_correct_color = np.logical_or(hsv_im[:, : , 0] > color_range[1], hsv_im[:, : , 0] < color_range[0])
            non_ball = np.logical_or(is_not_a_color, is_not_correct_color)

            single_color_img =  hsv_im.copy()
            single_color_img[non_ball] *= 0
            debug_msg = self.bridge.cv2_to_imgmsg(cv.cvtColor(single_color_img, cv.COLOR_HSV2RGB), encoding='rgb8')
            debug_msg.header = img_msg.header
            self.color_debuggers[color].publish(debug_msg)

            valid_points = np.logical_not(non_ball).nonzero()
            centers.append(self.calculate_blob_center(valid_points, h, w))

        draw_colors = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0)}
        for center, color in zip(centers, self.colors):

            if center is None:
                center = self.prev_known_centers[color]
            else:
                self.prev_known_centers[color] = center
            cv_image = cv.circle(cv_image, (int(center[1]), int(center[0])), 2, draw_colors[color], 1)

            coord_msg = Int16()
            coord_msg.data = int(center[1] - w/2)
            self.color_center_pubs[color].publish(coord_msg)

            
        #plt.title("Only largest blob points")
        #plt.scatter(border_points[:, 1], h-border_points[:, 0], color="red")
        #print("Results: {}".format(res.x))
        #plt.scatter([res.x[1]], [h - res.x[0]], color="blue")
        #plt.scatter([m_w], [h - m_h], color="green")

        # Show keypoints
        #cv.imshow("Keypoints", im_with_keypoints)
        # plt.show()
        #cv.waitKey(0)
        print("Publish results")
        #cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        result_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')
        result_msg.header = img_msg.header
        self.detection_publisher.publish(result_msg)


def main():

    rclpy.init()
    blob_detector = BlobDetector()
    rclpy.spin(blob_detector)
    blob_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
