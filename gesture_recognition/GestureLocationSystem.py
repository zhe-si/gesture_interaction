import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


class HandPoseModel:
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


class BodyPoseModel:
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
        self.point_pairs = [[1, 0], [1, 2], [1, 5],
                            [2, 3], [3, 4], [5, 6],
                            [6, 7], [1, 8], [8, 9],
                            [9, 10], [1, 11], [11, 12],
                            [12, 13], [0, 14], [0, 15],
                            [14, 16], [15, 17]]
        prototxt = os.path.join(
            modelpath,
            "coco/pose_deploy_linevec.prototxt")
        caffemodel = os.path.join(
            modelpath,
            "coco/pose_iter_440000.caffemodel")
        print(prototxt, caffemodel)
        coco_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return coco_model

    def general_body25_model(self, modelpath):
        self.num_points = 25
        self.point_pairs = [[1, 0], [1, 2], [1, 5],
                            [2, 3], [3, 4], [5, 6],
                            [6, 7], [0, 15], [15, 17],
                            [0, 16], [16, 18], [1, 8],
                            [8, 9], [9, 10], [10, 11],
                            [11, 22], [22, 23], [11, 24],
                            [8, 12], [12, 13], [13, 14],
                            [14, 19], [19, 20], [14, 21]]
        prototxt = os.path.join(
            modelpath,
            "body_25/pose_deploy.prototxt")
        caffemodel = os.path.join(
            modelpath,
            "body_25/pose_iter_584000.caffemodel")
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


class GestureLocationSystem:
    """
    手势定位系统
    通过opencv对手部进行定位
    1. 手部关键点提取
    """

    def __init__(self):
        pass


def main_hand():
    cap = cv2.VideoCapture(0)
    print("[INFO]Pose estimation.")

    # imgs_path = ""
    # os.listdir(imgs_path)
    # img_files = ['hand.jpg']
    start = time.time()
    model_path = "./model/openPose/hand"
    pose_model = HandPoseModel(model_path)
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
    pose_model = BodyPoseModel(model_path)
    print("[INFO]Model loads time: ", time.time() - start)

    for img_file in img_files:
        start = time.time()
        res_points = pose_model.predict(img_file)
        print("[INFO]Model predicts time: ", time.time() - start)
        pose_model.vis_pose(img_file, res_points)

    print("[INFO]Done.")


if __name__ == '__main__':
    # main_hand()
    main_body()
