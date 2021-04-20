import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from gesture_recognition import gesture_system


class HandOpenPoseModel:
    """
    OpenPose 手部关键点检查模型
    """

    def __init__(self, model_path):
        # 检测关键点数量
        self.num_points = 22
        # 手部关键点检测模型共输出22个关键点，其中包括手部的21个点，第22个点表示背景
        self.point_pairs = [[0, 1], [1, 2], [2, 3], [3, 4],
                            [0, 5], [5, 6], [6, 7], [7, 8],
                            [0, 9], [9, 10], [10, 11], [11, 12],
                            [0, 13], [13, 14], [14, 15], [15, 16],
                            [0, 17], [17, 18], [18, 19], [19, 20]]
        self.inHeight = 368
        self.threshold = 0.1
        self.hand_net = self.get_hand_model(model_path)

    # 模型加载
    @staticmethod
    def get_hand_model(model_path):
        proto_txt = os.path.join(model_path, "pose_deploy.prototxt")
        caffe_model = os.path.join(model_path, "pose_iter_102000.caffemodel")
        hand_model = cv2.dnn.readNetFromCaffe(proto_txt, caffe_model)

        return hand_model

    # 预测
    def predict(self, img_file, read_file=True):
        if read_file:
            img_cv2 = cv2.imread(img_file)
        else:
            img_cv2 = img_file
        img_height, img_width, _ = img_cv2.shape
        aspect_ratio = img_width / img_height

        in_width = int(((aspect_ratio * self.inHeight) * 8) // 8)
        inp_blob = cv2.dnn.blobFromImage(img_cv2, 1.0 / 255, (in_width, self.inHeight), (0, 0, 0), swapRB=False,
                                         crop=False)

        self.hand_net.setInput(inp_blob)

        output = self.hand_net.forward()

        # vis heatmaps
        self.vis_heatmaps(img_file, output, read_file)

        points = []
        for idx in range(self.num_points):
            prob_map = output[0, idx, :, :]  # confidence map.
            prob_map = cv2.resize(prob_map, (img_width, img_height))

            # Find global maxima of the probMap.
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            if prob > self.threshold:
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)

        return points

    # heatmap可视化
    def vis_heatmaps(self, img_file, net_outputs, read_file=True):
        if read_file:
            img_cv2 = cv2.imread(img_file)
        else:
            img_cv2 = img_file
        plt.figure(figsize=[10, 10])

        for pdx in range(self.num_points):
            prob_map = net_outputs[0, pdx, :, :]
            prob_map = cv2.resize(prob_map, (img_cv2.shape[1], img_cv2.shape[0]))
            plt.subplot(5, 5, pdx + 1)
            plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
            plt.imshow(prob_map, alpha=0.6)

            plt.colorbar()
            plt.axis("off")
        plt.show()

    # 手部关键点可视化
    def vis_pose(self, img_file, points, read_file=True):
        if read_file:
            img_cv2 = cv2.imread(img_file)
        else:
            img_cv2 = img_file
        img_cv2_copy = np.copy(img_cv2)
        for idx in range(len(points)):
            if points[idx]:
                cv2.circle(img_cv2_copy, points[idx], 8, (0, 255, 255), thickness=-1,
                           lineType=cv2.FILLED)
                cv2.putText(img_cv2_copy, "{}".format(idx), points[idx], cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2, lineType=cv2.LINE_AA)

        # 绘制连接点
        for pair in self.point_pairs:
            part_a = pair[0]
            part_b = pair[1]

            if points[part_a] and points[part_b]:
                cv2.line(img_cv2, points[part_a], points[part_b], (0, 255, 255), 3)
                cv2.circle(img_cv2, points[part_a], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        plt.figure(figsize=[10, 10])

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_cv2_copy, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()


class BodyOpenPoseModel:
    """
    OpenPose 人体识别
    """

    def __init__(self, modelpath, mode="BODY25"):
        # 指定采用的模型
        #   Body25: 25 points
        #   COCO:   18 points
        #   MPI:    15 points
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.1
        if mode == "BODY25":
            self.pose_net = self.general_body25_model(modelpath)
        elif mode == "COCO":
            self.pose_net = self.general_coco_model(modelpath)
        elif mode == "MPI":
            self.pose_net = self.get_mpi_model(modelpath)

    def get_mpi_model(self, modelpath):
        self.points_name = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14, "Background": 15}
        self.num_points = 15
        self.point_pairs = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8],
                            [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

        prototxt = os.path.join(modelpath, "mpi/pose_deploy_linevec_faster_4_stages.prototxt")
        caffemodel = os.path.join(modelpath, "mpi/pose_iter_160000.caffemodel")
        mpi_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return mpi_model

    def general_coco_model(self, modelpath):
        self.points_name = {
            "Nose": 0, "Neck": 1,
            "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7,
            "RHip": 8, "RKnee": 9, "RAnkle": 10,
            "LHip": 11, "LKnee": 12, "LAnkle": 13,
            "REye": 14, "LEye": 15,
            "REar": 16, "LEar": 17,
            "Background": 18}
        self.num_points = 18
        self.point_pairs = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
                            [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]
        prototxt = os.path.join(modelpath, "coco/pose_deploy_linevec.prototxt")
        caffemodel = os.path.join(modelpath, "coco/pose_iter_440000.caffemodel")
        print(prototxt, caffemodel)
        coco_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return coco_model

    def general_body25_model(self, modelpath):
        self.num_points = 25
        self.point_pairs = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6],
                            [6, 7], [0, 15], [15, 17], [0, 16], [16, 18], [1, 8],
                            [8, 9], [9, 10], [10, 11], [11, 22], [22, 23], [11, 24],
                            [8, 12], [12, 13], [13, 14], [14, 19], [19, 20], [14, 21]]
        prototxt = os.path.join(modelpath, "body_25/pose_deploy.prototxt")
        caffemodel = os.path.join(modelpath, "body_25/pose_iter_584000.caffemodel")
        coco_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return coco_model

    def predict(self, imgfile):
        img_cv2 = cv2.imread(imgfile)
        img_height, img_width, _ = img_cv2.shape
        inpBlob = cv2.dnn.blobFromImage(img_cv2,
                                        1.0 / 255,
                                        (self.inWidth, self.inHeight),
                                        (0, 0, 0),
                                        swapRB=False,
                                        crop=False)
        self.pose_net.setInput(inpBlob)
        self.pose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.pose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        output = self.pose_net.forward()

        H = output.shape[2]
        W = output.shape[3]
        print(output.shape)

        # vis heatmaps
        self.vis_heatmaps(imgfile, output)

        #
        points = []
        for idx in range(self.num_points):
            probMap = output[0, idx, :, :]  # confidence map.

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (img_width * point[0]) / W
            y = (img_height * point[1]) / H

            if prob > self.threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)

        return points

    def vis_heatmaps(self, imgfile, net_outputs):
        img_cv2 = cv2.imread(imgfile)
        plt.figure(figsize=[10, 10])

        for pdx in range(self.num_points):
            probMap = net_outputs[0, pdx, :, :]
            probMap = cv2.resize(
                probMap,
                (img_cv2.shape[1], img_cv2.shape[0])
            )
            plt.subplot(5, 5, pdx + 1)
            plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
            plt.imshow(probMap, alpha=0.6)
            plt.colorbar()
            plt.axis("off")
        plt.show()

    def vis_pose(self, imgfile, points):
        img_cv2 = cv2.imread(imgfile)
        img_cv2_copy = np.copy(img_cv2)
        for idx in range(len(points)):
            if points[idx]:
                cv2.circle(img_cv2_copy,
                           points[idx],
                           3,
                           (0, 0, 255),
                           thickness=-1,
                           lineType=cv2.FILLED)
                cv2.putText(img_cv2_copy,
                            "{}".format(idx),
                            points[idx],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .6,
                            (0, 255, 255),
                            1,
                            lineType=cv2.LINE_AA)

        # Draw Skeleton
        for pair in self.point_pairs:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(img_cv2,
                         points[partA],
                         points[partB],
                         (0, 255, 0), 3)
                cv2.circle(img_cv2,
                           points[partA],
                           3,
                           (0, 0, 255),
                           thickness=-1,
                           lineType=cv2.FILLED)
        cv2.imwrite("E:/1.jpg", img_cv2)
        cv2.imwrite("E:/2.jpg", img_cv2_copy)


class HandLocation:
    """
    手部定位与手指定位
    基于传统算法：肤色和轮廓
    """

    video_sta_shape = (176, 100)

    def __init__(self):
        pass

    @staticmethod
    def find_possible_move_hand(pic_bgr, flow_u, flow_v):
        pic_hsv = cv2.cvtColor(pic_bgr, cv2.COLOR_BGR2HSV)

        # 根据光流得到移动的区域
        thresh_u1, thresh_u2 = HandLocation.cal_move_bin(flow_u)
        thresh_v1, thresh_v2 = HandLocation.cal_move_bin(flow_v)
        move_mask = thresh_u1 + thresh_u2 + thresh_v1 + thresh_v2

        # 根据肤色二值化
        min_h, min_s, min_v = 0, 58, 0
        max_h, max_s, max_v = 50, 173, 255
        pic_bin = cv2.inRange(pic_hsv, (min_h, min_s, min_v), (max_h, max_s, max_v))

        # 将移动部分与肤色部分合成，确定手部位置
        pic_bin_move = cv2.bitwise_and(pic_bin, move_mask)

        # 去噪，填补漏洞
        blur_size, element_size = 5, 5
        pic_bin_move = cv2.medianBlur(pic_bin_move, blur_size)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2 * element_size + 1, 2 * element_size + 1), (element_size, element_size))
        pic_bin_move = cv2.dilate(pic_bin_move, element)

        # 确定手部的大致轮廓
        cnts, hierarchy = cv2.findContours(pic_bin_move, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return cnts, hierarchy

    @staticmethod
    def cal_move_bin(pic_gray, range_size=10):
        bg_col = stats.mode(pic_gray.reshape((pic_gray.shape[0] * pic_gray.shape[1])))[0][0]
        bg_range = (bg_col - range_size, bg_col + range_size)
        ret, thresh1 = cv2.threshold(pic_gray, bg_range[1], 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(pic_gray, bg_range[0], 255, cv2.THRESH_BINARY_INV)
        return thresh1, thresh2

    @staticmethod
    def cal_center(contour):
        contour_map = cv2.moments(contour)  # 计算第一条轮廓的各阶矩,字典形式
        center_x = int(contour_map["m10"] / contour_map["m00"])
        center_y = int(contour_map["m01"] / contour_map["m00"])
        return center_x, center_y

    @staticmethod
    def cal_power_area(contours, power_weight=0.6):
        if len(contours) == 0:
            return []
        area_list = []
        for i in range(len(contours)):
            area_list.append((i, cv2.contourArea(contours[i])))
        max_area = max(area_list, key=lambda x: x[1])
        min_area = max_area[1] * power_weight
        return [x[0] for x in area_list if x[1] > min_area]

    @staticmethod
    def get_tracker(tracker_type="CSRT"):
        if tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()  # 效果较好, 36ms, 图像缩小为标准后27ms
        elif tracker_type == "KCF":
            return cv2.TrackerKCF_create()  # 问题较大，20ms
        else:
            raise Exception("unknown tracker type")

        # tracker = cv2.TrackerGOTURN_create()  # 网上评价效果和kcf差不多，甚至不如
        # tracker = cv2.TrackerTLD_create()
        # tracker = cv2.TrackerMedianFlow_create()
        # tracker = cv2.TrackerMOSSE_create()

    @staticmethod
    def draw_bbox_in_pic(bbox, pic, color=(255, 0, 0), thickness=1, line_type=1, shift=None):
        """
        根据bbox在图上绘制矩形框

        Args:
            bbox: (start_p_x, start_p_y, size_x, size_y)
            pic: picture
            color: 矩形框颜色
            thickness: 线的宽度
            line_type: 线的类型
            shift: Number of fractional bits in the point coordinates.
        """
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(pic, p1, p2, color=color, thickness=thickness, lineType=line_type, shift=shift)

    @staticmethod
    def get_bbox_center(bbox):
        """
        得到bbox的中心点

        Args:
            bbox: (start_p_x, start_p_y, size_x, size_y)
        """
        return bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2


class GestureLocationSystem:
    """
    手势定位系统
    通过opencv对手部进行定位
    1. 手部关键点提取
    """

    def __init__(self, sta_shape=(176, 100)):
        self.sta_shape = sta_shape

        self.buffer_size = 15
        # 保存所有可能为手部的轮廓和中心点的信息
        self.hands_data_list = []
        # 保存跟踪为手部的轮廓和中心点的信息(name -> des, unsure_num) (des为list(contour, center))
        self.hands_map = {}
        # 暂未识别的手的信息
        self.unknown_hands_data = []

        # 两手距离历史记录
        self.latest_two_hands_dis_his = []
        # 单手面积历史记录
        self.latest_one_hand_area_his = []

        self.tracker = None

    @staticmethod
    def find_hands(flow_u_1, flow_v_1, pic_bgr_2):
        s = time.time()
        contours, hes = HandLocation.find_possible_move_hand(pic_bgr_2, flow_u_1, flow_v_1)
        tar_ids = HandLocation.cal_power_area(contours)
        hands_contour_center = []
        for tar_id in tar_ids:
            center_x, center_y = HandLocation.cal_center(contours[tar_id])
            hands_contour_center.append((contours[tar_id], (center_x, center_y)))
        print("calculate hand point use time: {}".format(time.time() - s))
        return hands_contour_center

    def get_hands_data(self, index=0):
        """得到手部对应的最新坐标（返回：name: point）"""
        hands_latest_data = {}
        for hand_name, des_list in self.hands_map.items():
            des_list = des_list[0]
            if len(des_list) > index:
                hands_latest_data[hand_name] = des_list[-1 - index]
        return hands_latest_data

    def get_hand_data_list(self, hand_name):
        """根据手的名字返回手的信息，若该名字对应的手不存在，返回None"""
        data = self.hands_map.get(hand_name, None)
        if data is None:
            return None
        return data[0]

    def get_hands_num(self):
        return len(self.hands_map.keys())

    def get_two_hands_distance(self, hands_name=None):
        """在指定了两个手或场上只有两个手时返回二者距离"""
        if hands_name is None:
            all_hands_name = list(self.hands_map.keys())
            hands_name = all_hands_name
        else:
            for hand_name in hands_name:
                if hand_name not in self.hands_map.keys():
                    return None
        if len(hands_name) != 2:
            return None
        hand1_data = self.hands_map[hands_name[0]][0]
        hand2_data = self.hands_map[hands_name[1]][0]
        if len(hand1_data) < 1 or len(hand2_data) < 1:
            return None
        return hands_name, self._get_distance(hand1_data[-1][1], hand2_data[-1][1])

    def get_one_hand_move_vector(self, hand_name=None):
        """在只有一个手的情况下返回其移动向量"""
        if hand_name is None:
            hands_name = list(self.hands_map.keys())
            if len(hands_name) != 1:
                return None
            hand_name = hands_name[0]
        else:
            if hand_name not in self.hands_map:
                return None
        hand_data = self.hands_map[hand_name][0]
        if len(hand_data) < 2:
            return None
        return hand_name, self.get_vector(hand_data[-2][1], hand_data[-1][1])

    def get_one_hand_area(self, hand_name=None):
        if hand_name is None:
            hands_name = list(self.hands_map.keys())
            if len(hands_name) != 1:
                return None
            hand_name = hands_name[0]
        else:
            if hand_name not in self.hands_map:
                return None
        hand_data = self.hands_map[hand_name][0]
        if len(hand_data) < 1:
            return None
        return hand_name, cv2.contourArea(hand_data[-1][0])

    def get_one_hand_area_list(self):
        return [data[1] for data in self.latest_one_hand_area_his]

    def get_two_hands_dis_list(self):
        return [data[1] for data in self.latest_two_hands_dis_his]

    @staticmethod
    def get_vector(point1, point2):
        return point2[0] - point1[0], point2[1] - point1[1]

    def _update_now_hands(self):
        """
        更新当前图片中的手及位置
        """
        new_hands_data = self._update_tailed_hands_sample()
        self._update_new_hands(new_hands_data)

    def _update_tailed_hands_tracker(self):
        """
        更新跟踪中的手的位置（暂未实现）

        使用opencv跟踪器，比较耗时
        """
        if self.tracker is None:
            tracker = HandLocation.get_tracker()
        pass

    def _update_tailed_hands_sample(self) -> list:
        """
        更新跟踪中的手的位置
        按照最近移动原则，速度较快
        """
        # 最新识别到的n个手位置的列表拷贝
        hands_data = self.hands_data_list[-1].copy()
        # 更新识别到的手的信息
        for hand_name, des_list_r in self.hands_map.items():
            des_list, unsure_num = des_list_r
            if len(des_list) > self.buffer_size:
                des_list.pop(0)
            if len(des_list) < 1:
                raise Exception("hand {} des_list is empty".format(hand_name))
            latest_des = des_list[-1]
            # 如果只有一拍信息，取阈值范围内最近的
            max_dis = self.sta_shape[1] / 3
            if len(des_list) == 1:
                min_dis, min_id = self._get_min_dis_id(hands_data, latest_des[1])
                if min_id != -1 and max_dis > min_dis:
                    des_list.append(hands_data.pop(min_id))
                    des_list_r[1] = 0
                else:
                    des_list_r[1] += 1
            # 如果有两拍以上数据，进行预测后取阈值范围内的
            else:
                start_des = des_list[-2]
                pre_point = self._get_pre_point(start_des[1], latest_des[1])
                if self._get_distance(pre_point, latest_des[1]) > self.sta_shape[1] / 20:
                    max_dis = self._get_distance(start_des[1], latest_des[1]) * 4
                else:
                    max_dis = self.sta_shape[1] / 4
                min_dis, min_id = self._get_min_dis_id(hands_data, pre_point)
                if min_id != -1 and max_dis > min_dis:
                    des_list.append(hands_data.pop(min_id))
                    des_list_r[1] = 0
                else:
                    des_list_r[1] += 1

        # 去除长时间不存在的手（判断为手离开，删除手的信息）
        left_hands = []
        unknown_to_left_num = 6
        for hand_name, des_list_r in self.hands_map.items():
            if des_list_r[1] > unknown_to_left_num:
                left_hands.append(hand_name)
        for hand_name in left_hands:
            self.hands_map.pop(hand_name)

        return hands_data

    def _get_min_dis_id(self, hands_data, latest_point):
        min_id, min_dis = -1, self.sta_shape[0] * 16
        for i in range(len(hands_data)):
            dis = self._get_distance(hands_data[i][1], latest_point)
            if dis < min_dis:
                min_id, min_dis = i, dis
        return min_dis, min_id

    @staticmethod
    def _get_distance(point1, point2):
        return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5

    @staticmethod
    def _get_pre_point(point1, point2):
        return point2[0] + point2[0] - point1[0], point2[1] + point2[1] - point1[1]

    def _update_new_hands(self, new_hands_data: list):
        """
        更新识别到的新的手
        """
        self.unknown_hands_data.append(new_hands_data)
        if len(self.unknown_hands_data) >= 3:
            known_data_ids = []
            for i in range(len(self.unknown_hands_data[-1])):
                now_path = []
                if self._tran(self.unknown_hands_data, -1, i, 1, 2, self.sta_shape[1] / 4, now_path):
                    self.hands_map[self._get_new_hand_name()] = [[self.unknown_hands_data[-1][i]], 0]
                    known_data_ids.append(i)
                    for j in range(len(now_path)):
                        self.unknown_hands_data[-2 - j].pop(now_path[j])
            known_data_ids.sort(reverse=True)
            for data_id in known_data_ids:
                self.unknown_hands_data[-1].pop(data_id)

    def _tran(self, ls: list, l1_id: int, l1_point_id: int, now_deep: int, tar_deep: int, tar_max_dis, now_path: list):
        if now_deep > tar_deep:
            return True
        p = ls[l1_id][l1_point_id][1]
        l2 = ls[l1_id - 1]
        min_id, min_dis = -1, 99999999
        for i in range(len(l2)):
            dis = self._get_distance(p, l2[i][1])
            if dis < min_dis:
                min_id, min_dis = i, dis
        if min_dis > tar_max_dis:
            return False
        now_path.append(min_id)
        return self._tran(ls, l1_id - 1, min_id, now_deep + 1, tar_deep, tar_max_dis, now_path)

    @staticmethod
    def _get_new_hand_name() -> str:
        return str(time.time())

    def _update_one_hand_area(self):
        if len(self.latest_one_hand_area_his) == 0:
            hand_area_data = self.get_one_hand_area()
            if hand_area_data is not None:
                self.latest_one_hand_area_his.append(hand_area_data)
        else:
            hand_area_data = self.get_one_hand_area(self.latest_one_hand_area_his[-1][0])
            if hand_area_data is not None:
                self.latest_one_hand_area_his.append(hand_area_data)
                if len(self.latest_one_hand_area_his) > self.buffer_size:
                    self.latest_one_hand_area_his.pop(0)
            else:
                self.latest_one_hand_area_his.clear()
                hand_area_data = self.get_one_hand_area()
                if hand_area_data is not None:
                    self.latest_one_hand_area_his.append(hand_area_data)

    def _update_two_hands_dis(self):
        if len(self.latest_two_hands_dis_his) == 0:
            hands_dis_data = self.get_two_hands_distance()
            if hands_dis_data is not None:
                self.latest_two_hands_dis_his.append(hands_dis_data)
        else:
            hands_dis_data = self.get_two_hands_distance(self.latest_two_hands_dis_his[-1][0])
            if hands_dis_data is not None:
                self.latest_two_hands_dis_his.append(hands_dis_data)
                if len(self.latest_two_hands_dis_his) > self.buffer_size:
                    self.latest_two_hands_dis_his.pop(0)
            else:
                self.latest_two_hands_dis_his.clear()
                hands_dis_data = self.get_two_hands_distance()
                if hands_dis_data is not None:
                    self.latest_two_hands_dis_his.append(hands_dis_data)

    def push_hand_data(self, hands_data):
        self.hands_data_list.append(hands_data)
        self._update_now_hands()
        self._update_one_hand_area()
        self._update_two_hands_dis()


def main_hand():
    video_sta_shape = (176, 100)

    cap = cv2.VideoCapture(0)
    print("[INFO]Pose estimation.")

    # imgs_path = ""
    # os.listdir(imgs_path)
    # img_files = ['hand.jpg']
    start = time.time()
    model_path = "./model/openPose/hand"
    pose_model = HandOpenPoseModel(model_path)
    print("[INFO]Model loads time: ", time.time() - start)

    # for img_file in img_files:
    while True:
        success, frame = cap.read()
        if success:
            start = time.time()
            res_points = pose_model.predict(frame, read_file=False)
            print("[INFO]Model predicts time: ", time.time() - start)
            pose_model.vis_pose(frame, res_points, read_file=False)
        else:
            print("camera read error")

    print("[INFO]Done.")


def main_body():
    # imgs_path = ""
    # os.listdir(imgs_path)
    img_files = ['E:/11.jpg']
    start = time.time()
    model_path = "./model/openPose/pose"
    pose_model = BodyOpenPoseModel(model_path)
    print("[INFO]Model loads time: ", time.time() - start)

    for img_file in img_files:
        start = time.time()
        res_points = pose_model.predict(img_file)
        print("[INFO]Model predicts time: ", time.time() - start)
        pose_model.vis_pose(img_file, res_points)

    print("[INFO]Done.")


def main_location():
    video_sta_shape = (176, 100)

    cap = cv2.VideoCapture(0)
    frame_bgr_list = []
    frame_gray_list = []
    flow_u_list = []
    flow_v_list = []
    ges_loc_sys = GestureLocationSystem()
    ii = 0
    while True:
        ii += 1
        success, frame = cap.read()
        if success:
            frame_bgr_list.append(cv2.resize(frame, video_sta_shape))
            frame_gray_list.append(cv2.cvtColor(frame_bgr_list[-1], cv2.COLOR_BGR2GRAY))
            if len(frame_bgr_list) > 15:
                frame_bgr_list.pop(0)
            if len(frame_bgr_list) > 1:
                f_u, f_v = gesture_system.DealVideo.calculate_brox_flow_with_dll(frame_gray_list[-2],
                                                                                 frame_gray_list[-1])
                flow_u_list.append(f_u)
                flow_v_list.append(f_v)
                if len(flow_u_list) > 14:
                    flow_u_list.pop(0)
                    flow_v_list.pop(0)
            if len(frame_bgr_list) > 1 and len(flow_u_list) > 1:
                # show_hand_area(flow_u_list[-1], flow_v_list[-1], frame_bgr_list[-1])
                hands_data = GestureLocationSystem.find_hands(flow_u_list[-1], flow_v_list[-1], frame_bgr_list[-1])
                pic_bgr_2 = frame_bgr_list[-1].copy()
                ges_loc_sys.push_hand_data(hands_data)
                for values in ges_loc_sys.hands_map.values():
                    cv2.drawContours(pic_bgr_2, np.array(values[0][-1]), 0, 255, -1)  # 绘制轮廓，填充
                    cv2.circle(pic_bgr_2, values[0][-1][1], 7, 128, -1)  # 绘制中心点
                cv2.imshow("t", pic_bgr_2)
                cv2.waitKey(30)
        else:
            print("camera read error")
            break


def main_tracker():
    video_sta_shape = (176, 100)

    # tracker = cv2.TrackerKCF_create()  # 问题较大，20ms
    # tracker = cv2.TrackerGOTURN_create()  # 网上评价效果和kcf差不多，甚至不如
    tracker = cv2.TrackerCSRT_create()  # 效果较好, 36ms, 图像缩小为标准后27ms
    # tracker = cv2.TrackerTLD_create()
    # tracker = cv2.TrackerMedianFlow_create()
    # tracker = cv2.TrackerMOSSE_create()

    cap = cv2.VideoCapture(0)

    success, frame = cap.read()
    frame = cv2.resize(frame, video_sta_shape)
    t = time.time()
    # bbox (start_p_x, start_p_y, size_x, size_y)
    bbox = cv2.selectROI("show", frame, showCrosshair=True, fromCenter=False)
    ok = tracker.init(frame, bbox)
    print("tracker init use {}s".format(time.time() - t))

    while True:
        success, frame = cap.read()
        frame = cv2.resize(frame, video_sta_shape)
        if success:
            t = time.time()
            ok, bbox = tracker.update(frame)
            print("tracker update use {}s".format(time.time() - t))
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                            2)
            cv2.imshow("show", frame)
            cv2.waitKey(20)
        else:
            print("camera read error")
            break
    pass


def show_hand_area(flow_u_1, flow_v_1, pic_bgr_2):
    s = time.time()
    contours, hes = HandLocation.find_possible_move_hand(pic_bgr_2, flow_u_1, flow_v_1)
    tar_ids = HandLocation.cal_power_area(contours)
    for tar_id in tar_ids:
        center_x, center_y = HandLocation.cal_center(contours[tar_id])
        cv2.drawContours(pic_bgr_2, contours, tar_id, 255, -1)  # 绘制轮廓，填充
        cv2.circle(pic_bgr_2, (center_x, center_y), 7, 128, -1)  # 绘制中心点
    print("calculate hand point use time: {}".format(time.time() - s))
    cv2.imshow("t", pic_bgr_2)
    cv2.waitKey(30)


if __name__ == '__main__':
    main_hand()
    # main_body()
    # main_location()
    # main_tracker()
