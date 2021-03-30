# 键盘手势视频，从n个文件夹拷贝中按照时间顺序每个拷贝m个视频到对应文件夹
import os
import shutil


def main():
    root_path = "E:/sourse/python/MFF-GestureRecognition/MFF-pytorch/datasets/click_keyboard"
    from_path1 = os.path.join(root_path, "video/lq电脑")
    from_path2 = os.path.join(root_path, "video/lq手机")
    from_path3 = os.path.join(root_path, "video/lzh电脑")
    from_path4 = os.path.join(root_path, "video/lzh手机")
    to_path = os.path.join(root_path, "datasets")
    make_sure_path_exits(to_path)

    video_list1 = get_file_list(from_path1)
    video_list2 = get_file_list(from_path2)
    video_list3 = get_file_list(from_path3)
    video_list4 = get_file_list(from_path4)
    for i in range(3, 3 * 25 - 1):
        left_right = "left"
        if i >= 5 * 25:
            left_right = "right"
        finger = i // 25 + 1
        point = i % 25 + 1
        dir_name = left_right + "-" + str(finger) + "-" + str(point)
        dir_name = os.path.join(to_path, dir_name)
        make_sure_path_exits(dir_name)
        if i == 10:
            n = 1
        else:
            n = 2
        move_n_file(from_path1, dir_name, video_list1, 2)
        move_n_file(from_path2, dir_name, video_list2, n)
        move_n_file(from_path3, dir_name, video_list3, 2)
        move_n_file(from_path4, dir_name, video_list4, 2)
    pass


# 从旧到新移动
def move_n_file(from_path, to_path, video_list, n: int):
    for i in range(n):
        file_name = video_list.pop(0)
        shutil.move(os.path.join(from_path, file_name), os.path.join(to_path, file_name))


def make_sure_path_exits(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_file_list(path: str):
    if not os.path.isdir(path):
        raise Exception("{} is not dir".format(path))
    file_list = os.listdir(path)
    file_list = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(path, x)))
    return file_list


if __name__ == '__main__':
    main()
