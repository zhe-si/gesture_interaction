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

    # test_file = open(r"E:\sourse\python\MFF-GestureRecognition\run_test\out\catch.txt", "w")

    all_time_sum = 0
    n = 0

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

                hand_vector_data = gesture_location_sys.get_one_hand_move_vector()
                hand_vector = None
                hand_name = None
                if hand_vector_data is not None:
                    hand_name, hand_vector = hand_vector_data

                package = {"gesture": [], "probability": [], "vector": hand_vector}

                # test_speed(hand_name, hand_vector, max_v, min_v, n, sum_v)

                for line in result:
                    package["gesture"].append(gesture_sys.categories_map[line[0]])
                    package["probability"].append(line[1].item())

                sender.send(package)
                print(package)

                # test_file.write(str(package))
                # test_file.write("\n")
                # test_file.flush()

                use_time = time.time() - s_t
                all_time_sum += use_time
                n += 1
                print("all use time {}s, avg: {}".format(use_time, all_time_sum / n))
                print()
            cv2.imshow("show", frame)
            cv2.waitKey(5)
        else:
            print("video read error")
            return


def test_speed(hand_name, hand_vector, max_v, min_v, n, sum_v):
    speed = -1
    if hand_vector is not None:
        speed = cal_vector_speed(*hand_vector)
        n += 1
        sum_v += speed
        if speed > max_v:
            max_v = speed
        if speed < min_v:
            min_v = speed
    if n > 0:
        print("hand: {} -> now: {}; max: {}; min: {}; avg: {}".format(
            hand_name, speed, max_v, min_v, sum_v / n))


def filter_label_by_location(output, ges_sys: GestureSystem, ges_loc_sys: GestureLocationSystem):
    """
    训练输出定位辅助筛选
    1. 滤去jester
    2. 根据手的数量

    3. 双手：判断turn的方向
    4. 双手：判断zoom是放大还是缩小
    5. 两手：根据两手距离判断turn和zoom
    6. 单手：移动快的swipe
    7. 单手：手轮廓变小catch；轮廓不变swipe和click
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

    hand_num = ges_loc_sys.get_hands_num()
    hand_vector_data = ges_loc_sys.get_one_hand_move_vector()
    hand_vector = None
    if hand_vector_data is not None:
        hand_name, hand_vector = hand_vector_data

    # 根据手的数量将正确的标签排到前面
    one_hand_label = ["Catch", "Click Down", "Click Up", "Pushing Two Fingers Away", "Shaking Hand", "Swipe",
                      "Thumb Up"]
    other_label = ["Doing other things"]
    two_hands_label = ["Turn With Two Hands Clockwise", "Turn With Two Hands Counterclockwise",
                       "Zooming In With Two Hands", "Zooming Out With Two Hands"]

    tar_output = check_hand_num(ges_sys, hand_num, one_hand_label, other_label, tar_output)
    jester_output = check_hand_num(ges_sys, hand_num, one_hand_label, other_label, jester_output)

    # 判断turn的方向
    if hand_num == 2:
        turn_clo_id, turn_counclo_id = None, None
        for i in range(len(tar_output)):
            if ges_sys.categories_map[tar_output[i][0]] == "Turn With Two Hands Clockwise":
                turn_clo_id = i
            elif ges_sys.categories_map[tar_output[i][0]] == "Turn With Two Hands Counterclockwise":
                turn_counclo_id = i
            if turn_clo_id is not None and turn_counclo_id is not None:
                break
        if turn_clo_id is not None and turn_counclo_id is not None:
            hands_data = ges_loc_sys.get_hands_data()
            hand_name1, hand_name2 = list(hands_data.keys())
            if hands_data[hand_name1][1][0] < hands_data[hand_name2][1][0]:
                top_hand_name = hand_name1
            else:
                top_hand_name = hand_name2
            if hands_data[hand_name1][1][1] < hands_data[hand_name2][1][1]:
                left_hand_name = hand_name1
            else:
                left_hand_name = hand_name2
            top_hand_vector = ges_loc_sys.get_one_hand_move_vector(hand_name=top_hand_name)
            left_hand_vector = ges_loc_sys.get_one_hand_move_vector(hand_name=left_hand_name)
            if top_hand_vector is not None and left_hand_vector is not None:
                top_hand_vector, left_hand_vector = top_hand_vector[1], left_hand_vector[1]
                if top_hand_vector[0] > 0 and left_hand_vector[1] < 0 and turn_counclo_id > turn_clo_id:
                    tar_output[turn_clo_id], tar_output[turn_counclo_id] = tar_output[turn_counclo_id], \
                                                                           tar_output[turn_clo_id]
                elif top_hand_vector[0] < 0 and left_hand_vector[1] > 0 and turn_counclo_id < turn_clo_id:
                    tar_output[turn_clo_id], tar_output[turn_counclo_id] = tar_output[turn_counclo_id], \
                                                                           tar_output[turn_clo_id]

    # 判断zoom是放大还是缩小
    if hand_num == 2:
        zoom_out_id, zoom_in_id = None, None
        for i in range(len(tar_output)):
            if ges_sys.categories_map[tar_output[i][0]] == "Zooming Out With Two Hands":
                zoom_out_id = i
            elif ges_sys.categories_map[tar_output[i][0]] == "Zooming In With Two Hands":
                zoom_in_id = i
            if zoom_out_id is not None and zoom_in_id is not None:
                break
        if zoom_out_id is not None and zoom_in_id is not None:
            hands_dis_his = ges_loc_sys.get_two_hands_dis_list()
            if len(hands_dis_his) > 1:
                # 放大
                if hands_dis_his[-1] > hands_dis_his[-2] and zoom_out_id > zoom_in_id:
                    tar_output[zoom_out_id], tar_output[zoom_in_id] = tar_output[zoom_in_id], tar_output[zoom_out_id]
                # 缩小
                elif hands_dis_his[-1] < hands_dis_his[-2] and zoom_out_id < zoom_in_id:
                    tar_output[zoom_out_id], tar_output[zoom_in_id] = tar_output[zoom_in_id], tar_output[zoom_out_id]

    # 若是两手, 判断两手距离：若相对不变，turn向前一格；若变化较大，zoom向前一格
    hands_dis_change_min = 27.2
    hands_dis_change_v_min = 10.0
    if hand_num == 2:
        hands_dis_his = ges_loc_sys.get_two_hands_dis_list()
        if len(hands_dis_his) > 1:
            check_num = 5
            hands_dis_list_tar = hands_dis_his[-min(check_num, len(hands_dis_his)):]
            hands_dis_all = abs(max(hands_dis_list_tar) - min(hands_dis_list_tar))
            hands_dis_now = abs(hands_dis_his[-1] - hands_dis_his[-2])

            # global sum_dis_all, n, sum_dis_now
            # sum_dis_all += hands_dis_all
            # sum_dis_now += hands_dis_now
            # n += 1
            # print("hands dis all: {}; avg: {}".format(hands_dis_all, sum_dis_all / n))
            # print("hands dis now: {}; avg: {}".format(hands_dis_now, sum_dis_now / n))

            if hands_dis_all > hands_dis_change_min or hands_dis_now > hands_dis_change_v_min:
                for i in range(1, len(tar_output)):
                    zoom_labels = ["Zooming In With Two Hands", "Zooming Out With Two Hands"]
                    if ges_sys.categories_map[tar_output[i][0]] in zoom_labels \
                            and ges_sys.categories_map[tar_output[i - 1][0]] not in zoom_labels:
                        tar_output[i - 1], tar_output[i] = tar_output[i], tar_output[i - 1]
            else:
                for i in range(1, len(tar_output)):
                    turn_labels = ["Turn With Two Hands Clockwise", "Turn With Two Hands Counterclockwise"]
                    if ges_sys.categories_map[tar_output[i][0]] in turn_labels \
                            and ges_sys.categories_map[tar_output[i - 1][0]] not in turn_labels:
                        tar_output[i - 1], tar_output[i] = tar_output[i], tar_output[i - 1]

    # 若是单手，判断移动速度，若大于阈值，swipe向前一格
    swipe_min_v = 8.2
    if hand_num == 1 and hand_vector is not None:
        swipe_speed = cal_vector_speed(*hand_vector)

        # global swipe_sum, n
        # swipe_sum += swipe_speed
        # n += 1
        # print("swipe speed: {}; avg: {}".format(swipe_speed, swipe_sum / n))

        if swipe_speed > swipe_min_v:
            for i in range(len(tar_output)):
                if ges_sys.categories_map[tar_output[i][0]] == "Swipe" and i != 0:
                    tar_output[i - 1], tar_output[i] = tar_output[i], tar_output[i - 1]
                    break

    # 若是单手，判断轮廓大小，若持续变小，catch向前一格；若持续不变，click和swipe向前一格（因为swipe出现较少，所以有两条提升条件）
    # 效果较差，作废
    # hands_area_influence(ges_loc_sys, ges_sys, hand_num, tar_output)

    return tar_output + jester_output


def hands_area_influence(ges_loc_sys, ges_sys, hand_num, tar_output):
    catch_hand_area_change_min = 5.0
    catch_hand_area_change_v_min = 1.0
    cl_sw_hand_area_change_max = 5.0
    cl_sw_hand_area_change_v_max = 1.0
    if hand_num == 1:
        hand_area_list = ges_loc_sys.get_one_hand_area_list()
        if len(hand_area_list) > 1:
            # 手总变化大于阈值
            check_num = 5
            hand_area_list_tar = hand_area_list[-min(check_num, len(hand_area_list)):]
            hands_area_all_diff = abs(max(hand_area_list_tar) - min(hand_area_list_tar))
            hands_area_now_diff = abs(hand_area_list[-1] - hand_area_list[-2])

            # global area_all_sum, area_now_sum, n
            # n += 1
            # area_now_sum += hands_area_now_diff
            # area_all_sum += hands_area_all_diff
            # print("area all sum: {}; avg: {}".format(hands_area_all_diff, area_all_sum / n))
            # print("area now sum: {}; avg: {}".format(hands_area_now_diff, area_now_sum / n))

            if hands_area_all_diff > catch_hand_area_change_min \
                    or hands_area_now_diff > catch_hand_area_change_v_min:
                for i in range(1, len(tar_output)):
                    if ges_sys.categories_map[tar_output[i][0]] == "Catch":
                        tar_output[i - 1], tar_output[i] = tar_output[i], tar_output[i - 1]
                        break
            elif hands_area_all_diff < cl_sw_hand_area_change_max \
                    or hands_area_now_diff < cl_sw_hand_area_change_v_max:
                for i in range(1, len(tar_output)):
                    no_move_label = ["Swipe", "Click Down", "Click Up"]
                    if ges_sys.categories_map[tar_output[i][0]] in no_move_label \
                            and ges_sys.categories_map[tar_output[i - 1][0]] not in no_move_label:
                        tar_output[i - 1], tar_output[i] = tar_output[i], tar_output[i - 1]


def cal_vector_speed(x, y):
    return (x ** 2 + y ** 2) ** 0.5


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
