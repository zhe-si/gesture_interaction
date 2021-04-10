# 模拟程序(剪切)
import cv2
import os
from copyVideo import make_sure_path_exits


def main():
    # video_path = r"E:\sourse\python\MFF-GestureRecognition\MFF-pytorch\datasets\keyboard_t\video\1.mp4"
    video_path = r"E:\sourse\python\MFF-GestureRecognition\MFF-pytorch\datasets\keyboard_t\video\2.mp4"

    to_path = r"E:\sourse\python\MFF-GestureRecognition\MFF-pytorch\datasets\keyboard_t\rgb\2"

    deal_video(video_path, to_path)


def deal_video(video_path: str, to_path: str):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cut_frame_num = 1
    now_num = 1
    step = 0
    while True:
        success, frame = cap.read()
        if success:
            if step % cut_frame_num == 0:
                pic_path = os.path.join(to_path, "%05d.jpg" % now_num).replace("\\", "/")
                if not cv2.imwrite(pic_path, cv2.resize(frame, (width // 2, height // 2))):
                    print("pic: {} > save failure".format(pic_path))
                now_num += 1
            step += 1
        else:
            break


if __name__ == '__main__':
    main()
