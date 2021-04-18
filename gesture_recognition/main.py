import time

from gesture_recognition.gesture_location_system import GestureLocationSystem
from gesture_recognition.gesture_system import GestureSystem
from gesture_recognition.transforms import *


def main():
    video_sta_shape = (176, 100)

    gesture_sys = GestureSystem("resnet101")
    # gesture_sys.load_model("./model/cc951_jester4_MFF_jester_RGBFlow_resnet101_segment5_3f1c_best.pth.tar")
    # gesture_sys.load_model("./model/cc961_lessjester_MFF_jester_RGBFlow_resnet101_segment5_3f1c_best.pth.tar")
    gesture_sys.load_model("./model/cc973_nojester_MFF_jester_RGBFlow_resnet101_segment5_3f1c_best.pth.tar")

    gesture_location_sys = GestureLocationSystem(video_sta_shape)

    # sender = Sender(Receiver.HOST, Receiver.PORT)

    test_file = open(r"E:\sourse\python\MFF-GestureRecognition\run_test\cc973_nojester\catch.txt", "w")

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        frame = cv2.resize(frame, video_sta_shape)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if success:
            s_t = time.time()
            result = gesture_sys.classify(frame_rgb)
            if result is not None:
                # print(gesture_sys.categories_map[result[0]])
                hands_data = GestureLocationSystem.find_hands(gesture_sys.deal_video.flow_u_list[-1],
                                                              gesture_sys.deal_video.flow_v_list[-1],
                                                              gesture_sys.deal_video.rgb_list[-1])
                gesture_location_sys.push_hand_data(hands_data)
                package = {"gesture": [], "probability": [], "vector": gesture_location_sys.get_hands_move_vector()}
                for line in result:
                    test_file.write(gesture_sys.categories_map[line[0]] + "; ")

                    print(gesture_sys.categories_map[line[0]] + "; ", end="")

                    package["gesture"].append(gesture_sys.categories_map[line[0]])
                    package["probability"].append(line[1])

                # sender.send(package)

                test_file.write("\n")
                test_file.flush()
                print()
                print("vector: {}".format(package["vector"]))
                print("all use time {}s".format(time.time() - s_t))
                print()
            cv2.imshow("show", frame)
            cv2.waitKey(5)
        else:
            print("video read error")
            return


def test(gesture_sys):
    pic_1 = cv2.imread("E:/test/1.jpg", cv2.IMREAD_GRAYSCALE)
    pic_2 = cv2.imread("E:/test/2.jpg", cv2.IMREAD_GRAYSCALE)
    pic_3 = cv2.imread("E:/test/3.jpg", cv2.IMREAD_GRAYSCALE)
    flow_u, flow_v = gesture_sys.deal_video.calculate_brox_flow_with_dll(pic_3, pic_2)
    cv2.imwrite("E:/test/f_u_n/1.jpg", flow_u)
    cv2.imwrite("E:/test/f_v_n/1.jpg", flow_v)
    flow_u, flow_v = gesture_sys.deal_video.calculate_brox_flow_with_dll(pic_2, pic_1)
    cv2.imwrite("E:/test/f_u_n/2.jpg", flow_u)
    cv2.imwrite("E:/test/f_v_n/2.jpg", flow_v)


if __name__ == '__main__':
    main()
