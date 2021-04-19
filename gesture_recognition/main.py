import time

from communication import Sender, Receiver
from gesture_recognition.gesture_location_system import GestureLocationSystem
from gesture_recognition.gesture_system import GestureSystem
from gesture_recognition.transforms import *


def main():
    video_sta_shape = (176, 100)

    gesture_sys = GestureSystem("resnet101")
    gesture_sys.load_model("./model/cc951_jester4_MFF_jester_RGBFlow_resnet101_segment5_3f1c_best.pth.tar")
    # gesture_sys.load_model("./model/cc961_lessjester_MFF_jester_RGBFlow_resnet101_segment5_3f1c_best.pth.tar")
    # gesture_sys.load_model("./model/cc973_nojester_MFF_jester_RGBFlow_resnet101_segment5_3f1c_best.pth.tar")

    gesture_location_sys = GestureLocationSystem(video_sta_shape)

    sender = Sender(Receiver.HOST, Receiver.PORT)

    test_file = open(r"E:\sourse\python\MFF-GestureRecognition\run_test\out\catch.txt", "w")

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        frame = cv2.resize(frame, video_sta_shape)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if success:
            s_t = time.time()
            # （标签序号，评估值），如果数据不足，会返回None
            result = gesture_sys.classify(frame_rgb)
            if result is not None:
                # print(gesture_sys.categories_map[result[0]])
                hands_data = GestureLocationSystem.find_hands(gesture_sys.deal_video.flow_u_list[-1],
                                                              gesture_sys.deal_video.flow_v_list[-1],
                                                              gesture_sys.deal_video.rgb_list[-1])
                gesture_location_sys.push_hand_data(hands_data)

                # 定位辅助判断
                result = filter_label_by_location(result, gesture_sys, gesture_location_sys)

                package = {"gesture": [], "probability": [], "vector": gesture_location_sys.get_hands_move_vector()}
                for line in result:
                    package["gesture"].append(gesture_sys.categories_map[line[0]])
                    package["probability"].append(line[1].item())

                sender.send(package)
                print(package)

                test_file.write(str(package))
                test_file.write("\n")
                test_file.flush()

                print("all use time {}s".format(time.time() - s_t))
                print()
            cv2.imshow("show", frame)
            cv2.waitKey(5)
        else:
            print("video read error")
            return


def filter_label_by_location(output, ges_sys: GestureSystem, ges_loc_sys: GestureLocationSystem):
    """
    训练输出定位辅助筛选
    1. 滤去jester
    2. 根据手的数量

    3. 两手：根据两手距离判断turn和zoom
    4. 单手：移动快的swipe
    5. 单手：手轮廓变小catch；轮廓不变swipe和click
    """
    # 将jester手势按顺序取出, 分成两个列表处理
    jester_label = ["Doing other things", "Pushing Two Fingers Away", "Shaking Hand", "Thumb Up"]
    tar_output = []
    jester_output = []
    for data in output:
        if ges_sys.categories_map[data[0]] in jester_label:
            jester_output.append(data)
        else:
            tar_output.append(data)

    # 根据手的数量将正确的标签排到前面
    one_hand_label = ["Catch", "Click Down", "Click Up", "Pushing Two Fingers Away", "Shaking Hand", "Swipe",
                      "Thumb Up"]
    other_label = ["Doing other things"]
    hand_num = ges_loc_sys.get_hands_num()

    tar_output = check_hand_num(ges_sys, hand_num, one_hand_label, other_label, tar_output)
    jester_output = check_hand_num(ges_sys, hand_num, one_hand_label, other_label, jester_output)

    # 若是两手, 判断两手距离：若相对不变，turn向前一格；若变化较大，zoom向前一格

    # 若是单手，判断移动速度，若大于阈值，swipe向前一格

    # 若是单手，判断轮廓大小，若持续变小，catch向前一格；若持续不变，click和swipe向前一格（因为swipe出现较少，所以

    return tar_output + jester_output


def check_hand_num(ges_sys, hand_num, one_hand_label, other_label, tar_output):
    output_tem = []
    output_tem_1 = []
    for data in tar_output:
        label = ges_sys.categories_map[data[0]]
        # 一个手或无手
        if hand_num <= 1:
            if label in one_hand_label or label in other_label:
                output_tem.append(data)
            else:
                output_tem_1.append(data)
        # 两个手、震荡多个手
        else:
            if label in one_hand_label or label in other_label:
                output_tem_1.append(data)
            else:
                output_tem.append(data)
    return output_tem + output_tem_1


if __name__ == '__main__':
    main()
