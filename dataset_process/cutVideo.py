# 播放视频根据输入剪切后另存为到标签对应路径
# 可将完整视频切成单个动作一个视频，并放到对应标签文件夹
# 空格继续，s + 标签名 开始某段视频，e + 标签名 结束某段视频，并将其保存到to_path的标签名文件夹里，开始和结束的标签名必须对应
import os
import time

import cv2

# 键盘按键：标签
sign_map = {"1": "ClickDown", "2": "ClickUp", "3": "Translation", "4": "ZoomOut",
            "5": "ZoomIn", "6": "Catch", "7": "Clockwise", "8": "Anticlockwise"}

time_format = "%Y-%m-%d-%H-%M-%S"


def main():
    # 保存路径
    to_path = "E:/1/2"
    # 视频路径(自动判断是单个视频文件还是一个文件夹)
    video_path = "E:/WIN_20210322_19_48_45_Pro.mp4"

    # 显示频率，空格跳过的时间/s，可自由设置
    show_interval = 0.1
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # mp4编节码器
    video_type = ".mp4"

    cut_video_auto(fourcc, show_interval, to_path, video_path, video_type)
    pass


def cut_video_auto(fourcc, show_interval, to_path, video_path, video_type):
    if os.path.isdir(video_path):
        cut_videos(fourcc, show_interval, to_path, video_path, video_type)
    elif os.path.isfile(video_path):
        cut_video(fourcc, show_interval, to_path, video_path, video_type)
    else:
        print("this is not a true path: {}".format(video_path))


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
    time_l = frame_num / fps
    show_frame = int(frame_num / (time_l / show_interval))
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
                            file_name = time.strftime(time_format, time.localtime(time.time()))
                            video_writer = \
                                cv2.VideoWriter(os.path.join(to_video_path, file_name + video_type),
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
