# 处理视频数据，转化为类似jester的数据集格式
# 1. 过滤过于小的视频
# 2. 把视频切成20张左右图片，放到对应文件夹，在csv中保存对应文件夹名和标签 (格式：34870;Drumming Fingers)
import os
import cv2
from copyVideo import make_sure_path_exits

video_root_path = r"F:\My_Resources\datasets\GestureRecognition_fwwb\video\根据标签切割"
# 要求to_path路径为英文路径
to_path = r"F:\My_Resources\datasets\GestureRecognition_fwwb\video\test"

save_video_path = "rgb"
data_label_file_name = "all.csv"

video_min_frame = 8
video_sta_shape = (176, 100)


def main():
    remove_data_label_file()
    deal_all_label()


def remove_data_label_file():
    data_label_file_path = os.path.join(to_path, data_label_file_name)
    if os.path.exists(data_label_file_path):
        os.remove(data_label_file_path)


def deal_all_label():
    label_list = os.listdir(video_root_path)
    label_list.remove(".finish")
    data_id = 1
    for label in label_list:
        label_path = os.path.join(video_root_path, label)
        video_list = os.listdir(label_path)
        video_id = 0
        make_sure_path_exits(to_path)
        with open(os.path.join(to_path, data_label_file_name), "a") as file:
            for video_name in video_list:
                video_path = os.path.join(label_path, video_name)
                deal_video(video_path, file, data_id, label)
                data_id += 1

                print("\r" + label + " > {} %".format(video_id / len(video_list)), end="")
                video_id += 1


def deal_video(video_path: str, file, data_id: int, label: str):
    cap = cv2.VideoCapture(video_path)
    frame_num = int(cap.get(7))
    if frame_num < video_min_frame:
        return False
    save_path = os.path.join(to_path, save_video_path, str(data_id))
    make_sure_path_exits(save_path)
    cut_frame_num = frame_num // 25
    cut_frame_num = cut_frame_num if cut_frame_num != 0 else 1
    now_num = 1
    step = 0
    while True:
        success, frame = cap.read()
        if success:
            if step % cut_frame_num == 0:
                pic_path = os.path.join(save_path, "%05d.jpg" % now_num).replace("\\", "/")
                if not cv2.imwrite(pic_path, cv2.resize(frame, video_sta_shape)):
                    print("pic: {} > save failure".format(pic_path))
                now_num += 1
            step += 1
        else:
            break
    file.write("{};{}\n".format(data_id, label))
    file.flush()


if __name__ == '__main__':
    main()
