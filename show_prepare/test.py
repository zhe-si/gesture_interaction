import time

import cv2
import numpy as np


# Freezing BatchNorm2D except the first one.
# calculate flow use time: 0.3332386016845703s
# calculate mff input use time: 0.400836706161499
# forward use time: 0.03123760223388672
# classify: Time: 0.432
# Doing other things, Thumb Up, Pushing Two Fingers Away, Shaking Hand, Click Down,

# old: 0.365433931350708s
# new: 0.09295892715454102s
def main():
    img1 = cv2.imread("E:/00012.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("E:/00013.jpg", cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread("E:/00014.jpg", cv2.IMREAD_GRAYSCALE)
    inst = cv2.optflow.createOptFlow_DeepFlow()
    s = time.time()
    flow = inst.calc(img2, img3, None)
    print(time.time() - s)
    cv2.imshow("test", show_flow_hsv(flow))
    cv2.waitKey(0)
    pass


def show_flow_hsv(flow, show_style=1):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # 将直角坐标系光流场转成极坐标系

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)

    # 光流可视化的颜色模式
    if show_style == 1:
        hsv[..., 0] = ang * 180 / np.pi / 2  # angle弧度转角度
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # magnitude归到0～255之间
    elif show_style == 2:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255

    # hsv转bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


if __name__ == '__main__':
    main()
