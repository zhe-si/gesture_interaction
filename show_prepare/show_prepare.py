# 展示效果的脚本
# 会显示n个窗口
# 1. 显示rgb图片
# 2. 显示光流的u
# 3. 显示光流的v
# 4. 显示当前敲击状态
# 5. 显示当前敲击位置状态
# 6. 输出字符

# 拍一个敲击n个字母的视频
# 生成每一帧的光流
# 显示rgb与光流，在一旁的窗口显示敲击状态与敲击位置，在一个txt中产生字符

import os
import cv2
from pynput.keyboard import Key, Controller


def main():
    data_id = 1
    fps = 30
    rgb_path = r"E:\sourse\python\MFF-GestureRecognition\MFF-pytorch\datasets\keyboard_t\rgb"
    flow_path = r"E:\sourse\python\MFF-GestureRecognition\MFF-pytorch\datasets\keyboard_t\flow"

    rgb_path = os.path.join(rgb_path, str(data_id))
    flow_u_path = os.path.join(flow_path, "u", str(data_id))
    flow_v_path = os.path.join(flow_path, "v", str(data_id))

    rgb_pics = os.listdir(rgb_path)
    rgb_pics.sort()
    rgb_pics.pop()
    flow_u_pics = os.listdir(flow_u_path)
    flow_u_pics.sort()
    flow_v_pics = os.listdir(flow_v_path)
    flow_v_pics.sort()

    l_down, l_up, l_other = "Down", "Up", "Other"
    time_up_down_map = {(59, 71): l_down, (72, 82): l_up, (113, 129): l_down, (130, 146): l_up, (186, 189): l_down, (190, 199): l_up,
                        (266, 277): l_down, (278, 287): l_up, (306, 316): l_down, (317, 328): l_up, (385, 398): l_down, (399, 421): l_up,
                        (455, 484): l_down, (490, 501): l_down, (502, 510): l_up, (569, 587): l_down, (588, 597): l_up, (652, 670): l_down, (671, 680): l_up,
                        (737, 755): l_down, (756, 767): l_up, (768, 770): l_down, (708, 822): l_down, (823, 833): l_up, (885, 898): l_down, (899, 916): l_up,
                        (954, 968): l_down, (969, 983): l_up, (1021, 1029): l_down, (1077, 1084): l_down, (1085, 1096): l_up, (1175, 1187): l_down, (1188, 1206): l_up,
                        (1262, 1279): l_down, (1280, 1294): l_up, (1342, 1368): l_down, (1369, 1382): l_up, (1483, 1494): l_down, (1495, 1507): l_up,
                        (1517, 1522): l_down, (1523, 1536): l_up, (1550, 1556): l_down, (1557, 1565): l_up, (1566, 1570): l_down, (1571, 1579): l_up,
                        (1606, 1613): l_down, (1614, 1627): l_up, (1670, 1691): l_down, (1692, 1708): l_up,
                        (1721, 1729): l_down, (1730, 1740): l_up, (1741, 1749): l_down, (1750, 1768): l_up, (1769, 1774): l_down, (1775, 1782): l_up,
                        (1790, 1795): l_down, (1796, 1814): l_up, (1900, 1914): l_down, (1915, 1931): l_up, (1969, 1974): l_down, (1975, 1982): l_up,
                        (1985, 2002): l_down, (2003, 2019): l_up, (2083, 2097): l_down, (2098, 2103): l_up, (2106, 2117): l_down, (2118, 2127): l_up,
                        (2128, 2136): l_down, (2137, 2143): l_up, (2144, 2153): l_down, (2154, 2164): l_up, (2165, 2171): l_down, (2172, 2179): l_up,
                        (2182, 2188): l_down, (2189, 2199): l_up, (2202, 2207): l_down, (2208, 2211): l_up, (2213, 2222): l_down, (2223, 2238): l_up,
                        (2244, 2253): l_down, (2254, 2262): l_up, (2277, 2287): l_down, (2288, 2301): l_up}
    time_click_point_map = {(57, 81): ("y", "right-2-7"), (116, 141): ("r", "left-4-8"), (185, 197): ("i", "right-3-8"),
                            (268, 281): ("j", "right-2-13"), (310, 327): ("g", "left-4-14"),
                            (394, 415): ("space", "left-5-13"), (495, 506): ("a", "left-1-13"), (579, 597): ("p", "right-5-8"),
                            (664, 676): ("w", "left-2-8"),
                            (744, 764): ("x", "left-2-18"), (718, 829): ("b", "left-4-19"), (890, 905): ("3", "left-3-3"), (963, 976): ("8", "right-3-3"),
                            (1033, 1041): ("0", "right-4-4"),
                            (1081, 1091): ("0", "right-4-4"), (1176, 1199): ("ctrl", "left-1-21"), (1269, 1285): ("shift", "left-1-17"),
                            (1350, 1379): ("shift", "left-1-17"),
                            (1489, 1502): ("h", "right-2-12"), (1518, 1529): ("e", "left-3-8"), (1552, 1563): ("l", "right-4-13"),
                            (1566, 1577): ("l", "right-4-13"), (1609, 1619): ("o", "right-4-8"),
                            (1687, 1695): ("w", "left-2-8"), (1724, 1737): ("o", "right-4-8"), (1745, 1764): ("r", "left-4-8"),
                            (1767, 1781): ("l", "right-4-13"), (1790, 1803): ("d", "left-3-13"),
                            (1906, 1924): ("space", "left-5-13"), (1970, 1981): ("l", "right-4-13"), (1997, 2008): ("q", "left-1-8"),
                            (2092, 2102): ("2", "left-2-3"), (2115, 2124): ("0", "right-4-4"),
                            (2131, 2141): ("1", "left-2-2"), (2148, 2160): ("8", "right-3-3"), (2168, 2180): ("3", "left-3-3"),
                            (2185, 2195): ("0", "right-4-4"),
                            (2203, 2211): ("3", "left-3-3"), (2217, 2228): ("0", "right-4-4"), (2249, 2257): ("2", "left-2-3"),
                            (2283, 2292): ("3", "left-3-3")}

    keyboard = Controller()
    for i in range(len(rgb_pics)):
        rgb_pic = cv2.imread(os.path.join(rgb_path, rgb_pics[i]))
        cv2.imshow("RGB input", rgb_pic)
        flow_u_pic = cv2.imread(os.path.join(flow_u_path, flow_u_pics[i]))
        cv2.imshow("Flow u input", flow_u_pic)
        flow_v_pic = cv2.imread(os.path.join(flow_v_path, flow_v_pics[i]))
        cv2.imshow("Flow v input", flow_v_pic)
        if i == 0:
            cv2.waitKey(1000 * 10)
        else:
            cv2.waitKey(30)
        frame_things(i, time_up_down_map, time_click_point_map, keyboard)


last_sign = "Other"


def frame_things(now_frame: int, t_ud_map: dict, t_cp_map: dict, keyboard):
    global last_sign
    l_ud, l_cp = "Other", "Other"
    now_click_point = "Other"
    # click point
    for key, sign in t_cp_map.items():
        if key[0] <= now_frame <= key[1]:
            print("click point sign: {}, map to {}".format(sign[1], sign[0]))
            now_click_point = sign[0]
            break
    else:
        print("click point sign: {}, map to {}".format(l_cp, None))
    # up or down
    for key, sign in t_ud_map.items():
        if key[0] <= now_frame <= key[1]:
            print("Up or Down sign: {}".format(sign))
            if last_sign == "Down" and sign == "Up" and now_click_point != "Other":
                if now_click_point == "space":
                    click_key(keyboard, Key.space)
                elif now_click_point == "ctrl":
                    click_key(keyboard, Key.ctrl)
                elif now_click_point == "shift":
                    click_key(keyboard, Key.shift)
                else:
                    click_key(keyboard, now_click_point)
            last_sign = sign
            break
    else:
        print("Up or Down sign: {}".format(l_ud))
        last_sign = "Other"


def click_key(keyboard, key):
    keyboard.press(key)
    keyboard.release(key)


if __name__ == '__main__':
    main()
