# 将视频处理为固定张数的数据
import random

import cv2
import os
from copyVideo import make_sure_path_exits


def main():
    videos_path = r"F:\My_Resources\datasets\click_keyboard\video\updown\other"
    to_path = r"F:\My_Resources\datasets\click_keyboard\updown_j"
    data_num = 16
    video_sta_shape = (176, 100)
    tar_videos_num = 800

    start_id = 1849
    label = "Other"

    videos_name = os.listdir(videos_path)
    videos_num = len(videos_name)
    videos_name = (tar_videos_num // videos_num) * videos_name + random.sample(videos_name,
                                                                               k=tar_videos_num % videos_num)
    now_point = 0
    with open(os.path.join(to_path, "all.csv"), "a") as all_csv:
        for video_name in videos_name:
            video_path = os.path.join(videos_path, video_name)
            deal_video_adjust(video_path, all_csv, start_id, label, to_path, data_num, video_sta_shape)
            start_id += 1
            now_point += 1
            print("\rready: {}%".format(now_point / tar_videos_num * 100), end="")


def deal_video_adjust(video_path: str, sign_file, data_id: int, label: str, to_path: str, data_num: int,
                      video_sta_shape):
    cap = cv2.VideoCapture(video_path)
    frame_num = int(cap.get(7))

    save_path = os.path.join(to_path, "rgb", str(data_id))
    make_sure_path_exits(save_path)
    now_num = 1
    now_frame = 1

    if data_num > frame_num:
        each_num = data_num // frame_num
        random_point = random.sample(list(range(1, frame_num + 1)), k=data_num % frame_num)
        random_point.sort()
        while True:
            success, frame = cap.read()
            if success:
                for i in range(each_num):
                    save_pic(frame, now_num, save_path, video_sta_shape)
                    now_num += 1
                if len(random_point) != 0:
                    if random_point[0] == now_frame:
                        random_point.pop(0)
                        save_pic(frame, now_num, save_path, video_sta_shape)
                        now_num += 1
                now_frame += 1
            else:
                break
    else:
        random_point = random.sample(list(range(1, frame_num + 1)), k=data_num)
        random_point.sort()
        while True:
            success, frame = cap.read()
            if success:
                if len(random_point) != 0:
                    if random_point[0] == now_frame:
                        random_point.pop(0)
                        save_pic(frame, now_num, save_path, video_sta_shape)
                        now_num += 1
                now_frame += 1
            else:
                break
    sign_file.write("{};{}\n".format(data_id, label))
    sign_file.flush()


def save_pic(frame, now_num, save_path, video_sta_shape):
    pic_path = os.path.join(save_path, "%05d.jpg" % now_num).replace("\\", "/")
    if not cv2.imwrite(pic_path, cv2.resize(frame, video_sta_shape)):
        print("pic: {} > save failure".format(pic_path))


if __name__ == '__main__':
    main()
