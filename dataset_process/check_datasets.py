# 检查是否有过少的文件并重新规范编号,同时可以计算平均图片张数
import os
import shutil
from copyVideo import make_sure_path_exits


def main():
    root_path = r"F:\My_Resources\datasets\click_keyboard\updown_j"
    data_path = os.path.join(root_path, "rgb")
    new_data_path = os.path.join(root_path, "rgb_new")
    make_sure_path_exits(new_data_path)
    sign_file = os.path.join(root_path, "all.csv")
    new_sign_file = os.path.join(root_path, "all_ok.csv")

    tar_min_data_num = 5
    data_num_map = {}

    only_check = True

    rgb_ids = os.listdir(data_path)
    new_id = 0
    with open(sign_file, "r") as old_file, open(new_sign_file, "w") as new_file:
        for line in old_file:
            data_id, label = line.split(";")
            if data_id not in rgb_ids:
                print("error: data id {} is wrong.".format(data_id))
                break
            tar_data_path = os.path.join(data_path, data_id)
            data_num = len(os.listdir(tar_data_path))
            if data_num < tar_min_data_num:
                print("wring: data id {}'s data num is less than {}".format(data_id, tar_min_data_num))
                continue
            if data_num in data_num_map:
                data_num_map[data_num] += 1
            else:
                data_num_map[data_num] = 1

            if not only_check:
                new_id += 1
                new_tar_data_path = os.path.join(new_data_path, str(new_id))
                new_file.write("{};{}".format(new_id, label))
                if os.path.exists(new_tar_data_path):
                    print("error: new data id {} is repeat".format(new_id))
                    break
                else:
                    shutil.copytree(tar_data_path, new_tar_data_path)
    print(data_num_map)
    print("avg: {}; min: {}; max: {};".format(*get_avg_min_max(data_num_map)))


def get_avg_min_max(tar_map: dict):
    all_now = 0
    min_value = 999999
    max_value = -1
    num = 0
    for key, value in tar_map.items():
        all_now += key * value
        num += value
        if key > max_value:
            max_value = key
        if key < min_value:
            min_value = key
    return all_now / num, min_value, max_value


if __name__ == '__main__':
    main()
