import cv2 as cv
import numpy as np
from collections import deque
import os

FOLDER_NAME = 'stereo_data'
CAM1_IMAGE_PREFIX = 'cam1_image'
CAM1_IMAGE_TYPE = '.jpg'
CAM2_IMAGE_PREFIX = 'cam2_image'
CAM2_IMAGE_TYPE = '.jpg'

cam1_img = deque()
cam2_img = deque()
cam1_points = deque()
cam2_points = deque()

cam1_array = np.array(None)
cam2_array = np.array(None)

cam1_fx = 204.64863681
cam1_fy = 204.47041377
cam1_cx = 308.78000754
cam1_cy = 258.21809417
cam1_k1 = 0.24406997
cam1_k2 = -0.22412072
cam1_p1 = -0.00079476
cam1_p2 = -0.00035923
cam1_k3 = 0.05262498

cam2_fx = 204.42765186
cam2_fy = 204.43521494
cam2_cx = 310.99781296
cam2_cy = 257.91267286
cam2_k1 = 0.2305133
cam2_k2 = -0.20287915
cam2_p1 = -0.00140612
cam2_p2 = 0.0033575
cam2_k3 = 0.04448097

cam1_matrix = np.array([[cam1_fx, 0, cam1_cx],
                          [0, cam1_fy, cam1_cy],
                          [0, 0, 1]], dtype=np.float64)
cam1_dist = np.array([cam1_k1, cam1_k2, cam1_p1, cam1_p2, cam1_k3], dtype=np.float64)

cam2_matrix = np.array([[cam2_fx, 0, cam2_cx],
                          [0, cam2_fy, cam2_cy],
                          [0, 0, 1]], dtype=np.float64)
cam2_dist = np.array([cam2_k1, cam2_k2, cam2_p1, cam2_p2, cam2_k3], dtype=np.float64)

def find_dot_from_image(img):
    # img = cv.GaussianBlur(img,(5,5),0)
    grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    grey = cv.threshold(grey, 255 * 0.9, 255, cv.THRESH_BINARY)[1]
    contours, _ = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(img, contours, -1, (0, 255, 0), 1)

    image_points = []
    for contour in contours:
        moments = cv.moments(contour)
        if moments["m00"] != 0:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            image_points.append([center_x, center_y])
            center_x = int(center_x)
            center_y = int(center_y)
            cv.putText(img, f'({center_x}, {center_y})', (center_x, center_y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                       (100, 255, 100), 1)
            cv.circle(img, (center_x, center_y), 1, (100, 255, 100), -1)

    if len(image_points) == 0:
        image_points = [[None, None]]

    return img, image_points


def processs_pair_image(img1, img2):
    img1, points1 = find_dot_from_image(img1)
    img2, points2 = find_dot_from_image(img2)
    count1 = len(points1)
    count2 = len(points2)
    # print(f"cam1:{count1}points cam2:{count2}points")
    # cv.imshow("img1", img1)
    # cv.imshow("img2", img2)
    # cv.waitKey(10)
    if count1 == 1 and count2 == 1:
        cam1_points.append(points1[0])
        cam2_points.append(points2[0])
        return
    else:
        return


def load_data():
    pair_count = 1
    while True:
        path1 = "./" + FOLDER_NAME + "/" + CAM1_IMAGE_PREFIX + f"{pair_count}" + CAM1_IMAGE_TYPE
        path2 = "./" + FOLDER_NAME + "/" + CAM2_IMAGE_PREFIX + f"{pair_count}" + CAM2_IMAGE_TYPE
        # print(path1)
        # print(path2)
        exit1 = os.path.exists(path1)
        exit2 = os.path.exists(path2)

        if exit1 and exit2:
            print(f"find pair image:{pair_count}")
            img1 = cv.imread(path1)
            img2 = cv.imread(path2)
            # cv.imshow("img1", img1)
            # cv.imshow("img2", img2)
            # cv.waitKey(50)
            cam1_img.append(img1)
            cam2_img.append(img2)

            pair_count += 1

        elif exit1 or exit2:
            print("one of the camera image is missing!")
            break
        else:
            print(f"Read done {pair_count-1} pairs")
            break

def process_images():
    while cam1_img and cam2_img:
        img1 = cam1_img.popleft()
        img2 = cam2_img.popleft()
        processs_pair_image(img1, img2)

def compute_essential_matrix():
    global cam1_array, cam2_array
    cam1_list = []
    cam2_list = []
    while cam1_points and cam2_points:
        point1 = cam1_points.popleft()
        point2 = cam2_points.popleft()
        cam1_list.append(point1)
        cam2_list.append(point2)

    cam1_array = np.array(cam1_list)
    cam2_array = np.array(cam2_list)

    # cam1_undist_array = cam1_get_normal_pixel(cam1_array)
    # cam2_undist_array = cam2_get_normal_pixel(cam2_array)
    # cam1_undist_array = cam1_undist_array.reshape(cam1_undist_array.shape[0], 2)
    # cam2_undist_array = cam2_undist_array.reshape(cam2_undist_array.shape[0], 2)
    # print(cam1_undist_array.shape)  # N*1*2
    # print(cam1_undist_array[0][0])  获取单个归一化像素坐标

    # 计算本质矩阵
    E, mask = cv.findEssentialMat(
        points1=cam1_array,
        points2=cam2_array,
        cameraMatrix=cam1_matrix,
        method=cv.RANSAC,
        threshold=3.0
    )
    print("the essential mask:")
    print(mask)
    print("the essential matrix:")
    print(E)
    ret, R, t, mask = cv.recoverPose(
        E=E,
        points1=cam1_array,
        points2=cam2_array,
        cameraMatrix=cam1_matrix
    )
    print("the point mask:")
    print(mask)
    print("the rotation matrix:")
    print(R)
    print("the t vector")
    print(t)
    return E


load_data()
process_images()
E = compute_essential_matrix()