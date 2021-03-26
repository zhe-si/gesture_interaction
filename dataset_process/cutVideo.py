# 播放视频根据输入剪切后另存为到标签对应路径
# 可将完整视频切成单个动作一个视频，并放到对应标签文件夹
import os

import cv2

# 键盘按键：标签
sign_map = {"1": "ClickDown", "2": "ClickUp", "3": "Translation", "4": "ZoomOut",
            "5": "ZoomIn", "6": "Catch", "7": "Clockwise", "8": "Anticlockwise"}


def main():
    # 保存路径
    to_path = "D:/document/服务外包大赛/视频手势/dealWithVideo"
    # 单个视频路径
    # video_path = "E:/WIN_20210322_19_46_16_Pro.mp4"

    # 视频文件夹路径
    videos_dir_path = "D:/document/服务外包大赛/视频手势/电脑"

    # 显示频率，空格跳过的时间/s，可自由设置
    show_interval = 0.1
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # mp4编节码器
    video_type = ".mp4"

    # 剪切单个视频
    # cut_video(fourcc, show_interval, to_path, video_path, video_type)
    # 剪切多个视频（videos_dir_path是视频文件夹路径）
    cut_videos(fourcc, show_interval, to_path, videos_dir_path, video_type)
    pass


# 剪切多个文件，输入一个原视频文件夹
def cut_videos(fourcc, show_interval, to_path, videos_dir_path, video_type):
    videos_path = os.listdir(videos_dir_path)
    for video_path in videos_path:
        path = os.path.join(videos_dir_path, video_path)
        if os.path.isfile(path):
            print("start cut video {}".format(video_path))
            cut_video(fourcc, show_interval, to_path, path, video_type)
            print("end cut video {}".format(video_path))


# 剪切单个视频
def cut_video(fourcc, show_interval, to_path, video_path, video_type):
    cap = cv2.VideoCapture(video_path)

    frame_num = int(cap.get(7))
    fps = int(cap.get(5))  # 获得视频帧速率
    time = frame_num / fps
    show_frame = int(frame_num / (time / show_interval))
    if show_frame < 1:
        show_frame = 1
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    i = 0
    video_index = 1
    sign_state = (" ", " ")
    video_writer = None
    while True:
        success, frame = cap.read()
        if success:
            i += 1
            # 每show_frame显示一次图片并判断是否
            if i % show_frame == 1 or show_frame == 1:
                cv2.imshow("show", frame)
                while True:
                    key = chr(cv2.waitKey(0))
                    # 跳过
                    if key == " ":
                        if video_writer is not None:
                            video_writer.write(frame)
                        break
                    # 开始某标签剪切
                    elif key == "s":
                        if sign_state[0] != " ":
                            print("last sign is not end")
                            continue
                        key2 = chr(cv2.waitKey(0))
                        if key2 not in sign_map:
                            print("don't have this sign, start sign false")
                        else:
                            to_video_path = os.path.join(to_path, sign_map[key2])
                            mack_sure_dir_exit(to_video_path)
                            video_writer = \
                                cv2.VideoWriter(os.path.join(to_video_path, str(video_index) + video_type),
                                                fourcc, fps, (width, height))
                            video_writer.write(frame)
                            sign_state = ("s", key2)
                            video_index += 1
                            print("start sign {}".format(sign_map[key2]))
                        continue
                    # 结束某标签剪切
                    elif key == "e":
                        if sign_state[0] != "s":
                            print("sign is not start")
                            continue
                        key2 = chr(cv2.waitKey(0))
                        if key2 != sign_state[1]:
                            print("end sign {} is not match start sign {}, end sign false"
                                  .format(key2, sign_map[sign_state[1]]))
                        else:
                            video_writer.write(frame)
                            video_writer.release()
                            video_writer = None
                            sign_state = (" ", " ")
                            print("end sign {}".format(sign_map[key2]))
                        continue
            else:
                if video_writer is not None:
                    video_writer.write(frame)
        else:
            print('end video {}'.format(video_path))
            break


def mack_sure_dir_exit(to_video_path):
    if not os.path.exists(to_video_path):
        os.makedirs(to_video_path)


if __name__ == '__main__':
    main()
