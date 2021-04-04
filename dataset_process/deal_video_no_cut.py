# 自动分割视频并转换为jester格式
import os
import random
import shutil
import time

import cv2

import deal_video
from copyVideo import make_sure_path_exits

start_cut_time = 25
end_cut_time = 70

# 切割份数
cut_num = 9
random_cut_num = cut_num // 2

root_labels_path = r"F:\My_Resources\datasets\click_keyboard\video\datasets"
to_path = r"F:\My_Resources\datasets\click_keyboard\output"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4编节码器
video_type = ".mp4"


def main():
    deal_video_with_cut(root_labels_path)
    pass


def deal_video_with_cut(labels_path: str):
    save_video_path = os.path.join(to_path, deal_video.save_video_path)
    make_sure_path_exits(save_video_path)
    data_id = get_max_id(save_video_path) + 1
    labels = os.listdir(labels_path)
    for label in labels:
        videos_path = os.path.join(labels_path, label)
        videos = os.listdir(videos_path)
        videos_num = len(videos)
        now_num = 0
        with open(os.path.join(to_path, deal_video.data_label_file_name), "a") as file:
            for video in videos:
                run_start_time = int(time.time())
                video_path = os.path.join(videos_path, video)
                tem_videos_path = os.path.join(videos_path, "..", str(int(time.time())))
                make_sure_path_exits(tem_videos_path)
                cut_video(video_path, tem_videos_path)
                cut_videos = os.listdir(tem_videos_path)
                for video_cut in cut_videos:
                    video_cut_path = os.path.join(tem_videos_path, video_cut)
                    deal_video.deal_video(video_cut_path, file, data_id, label, to_path)
                    data_id += 1
                shutil.rmtree(tem_videos_path)

                now_num += 1
                print("\r" + label + " > {}%  use: {}s".
                      format(now_num / videos_num * 100, int(time.time()) - run_start_time), end="")


def get_max_id(path: str):
    ids = [int(i) for i in os.listdir(path)]
    if len(ids) == 0:
        return 0
    else:
        return max(ids)


def cut_video(video_path: str, to_videos_path: str):
    cap = cv2.VideoCapture(video_path)
    frame_num = int(cap.get(7))
    fps = int(cap.get(5))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    save_id = 1
    cut_pairs = get_cut_pairs(frame_num)
    save_videos_map = {}
    for i in range(frame_num):
        success, frame = cap.read()
        if success:
            for pair in cut_pairs:
                if i == pair[0] or (pair[1] > i > pair[0] and pair not in save_videos_map):
                    save_videos_map[pair] = cv2.VideoWriter(os.path.join(to_videos_path, str(save_id) + video_type),
                                                            fourcc, fps, (width, height))
                    save_id += 1
                elif i == pair[1] and pair in save_videos_map:
                    save_videos_map[pair].release()
                    save_videos_map.pop(pair)
                elif i < pair[0]:
                    break
            for writer in save_videos_map.values():
                writer.write(frame)
        else:
            print("video read error !!!")
            break
    if len(save_videos_map.items()) != 0:
        print("error: save_videos_map writer manage error")
        for writer in save_videos_map.values():
            writer.release()


def get_cut_pairs(frame_num: int):
    cut_pairs = []
    std_cut_length = (frame_num - start_cut_time - end_cut_time) // cut_num
    if std_cut_length < 3:
        print("waring: std_cut_length < 3")
    if start_cut_time >= frame_num - end_cut_time:
        print("error: start_cut_time >= frame_num - end_cut_time")
        return cut_pairs
    for i in range(start_cut_time, frame_num - end_cut_time - std_cut_length, std_cut_length):
        random_end = i + std_cut_length + random.randint(-1, std_cut_length // 4)
        if random_end > frame_num - end_cut_time:
            random_end = i + std_cut_length
        cut_pairs.append((i, random_end))
    for i in range(random_cut_num):
        random_start = random.randint(start_cut_time, frame_num - end_cut_time - std_cut_length)
        random_length = std_cut_length + random.randint(-1, std_cut_length // 5)
        if random_start + random_length > frame_num - end_cut_time:
            random_length = std_cut_length
        cut_pairs.append((random_start, random_start + random_length))
    cut_pairs = list(set(cut_pairs))
    cut_pairs.sort(key=lambda x: x[0])
    return cut_pairs


if __name__ == '__main__':
    main()
