import ctypes
import time

from gesture_recognition.transforms import *


class DealVideo:
    # 加载计算DeepFlow计算器
    inst = cv2.optflow.createOptFlow_DeepFlow()

    # 加载计算光流dll
    dll_path = "./lib/flow_computation.dll"
    lib = ctypes.CDLL(dll_path, winmode=ctypes.RTLD_GLOBAL)
    lib.calulateFlow_kkk.restype = ctypes.POINTER(ctypes.c_uint8)

    def __init__(self, new_length, modality, normalize, model_name, scale_size, crop_size,
                 num_segments, transform=None, buffer_length=24):
        self.new_length = new_length
        self.modality = modality
        self.normalize = normalize
        self.scale_size = scale_size
        self.crop_size = crop_size
        self.num_segments = num_segments
        if transform is not None:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.Compose([
                GroupScale(int(self.scale_size)),
                GroupCenterCrop(self.crop_size),
                Stack(roll=(model_name in ['BNInception', 'InceptionV3']),
                      isRGBFlow=(self.modality == 'RGBFlow')),
                ToTorchFormatTensor(div=(model_name not in ['BNInception', 'InceptionV3'])),
                self.normalize,
            ])

        if self.modality == 'RGBDiff' or self.modality == 'RGBFlow':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self.min_num = self.new_length + self.num_segments - 1

        # 初始化图片缓冲区
        self.buffer_length = buffer_length
        self.rgb_list = []
        self.gray_list = []
        self.flow_u_list = []
        self.flow_v_list = []

    @staticmethod
    def _get_val_indices(num_frames, num_segments, new_length):
        if num_frames > num_segments + new_length - 1:
            tick = (num_frames - new_length + 1) / float(num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))
        return offsets + 1

    def set_buffer_length(self, length):
        self.buffer_length = length
        self.update_buffer()

    def push_new_rgb(self, rgb_img):
        self.rgb_list.append(rgb_img)
        self.gray_list.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY))
        if len(self.rgb_list) > 1:
            flow_u, flow_v = self.calculate_brox_flow_with_dll(self.gray_list[-2], self.gray_list[-1])
            self.flow_u_list.append(flow_u)
            self.flow_v_list.append(flow_v)
        self.update_buffer()

    def update_buffer(self):
        while len(self.rgb_list) > self.buffer_length:
            self.rgb_list.pop(0)
            self.gray_list.pop(0)
        while len(self.flow_u_list) > self.buffer_length - 1:
            self.flow_u_list.pop(0)
            self.flow_v_list.pop(0)

    def get_mff_input_from_buffer(self):
        if len(self.rgb_list) < self.min_num:
            return None
        indices = DealVideo._get_val_indices(len(self.rgb_list), self.num_segments, self.new_length)
        return self._turn_imgs_to_mff_input(self.rgb_list, self.flow_u_list, self.flow_v_list, indices)

    def turn_imgs_to_mff_input(self, rgb_list):
        indices = DealVideo._get_val_indices(len(rgb_list), self.num_segments, self.new_length)
        flow_u_list, flow_v_list = self.calculate_rgb_list_flow(rgb_list)
        return self._turn_imgs_to_mff_input(rgb_list, flow_u_list, flow_v_list, indices)

    def calculate_rgb_list_flow(self, rgb_list, mode="brox"):
        gray_list = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in rgb_list]
        flow_u_list = []
        flow_v_list = []
        for i in range(len(gray_list) - 1):
            if mode == "brox":
                flow_u, flow_v = self.calculate_brox_flow_with_dll(gray_list[i], gray_list[i + 1])
            elif mode == "deep_flow":
                flow_u, flow_v = self.calculate_deep_flow(gray_list[i], gray_list[i + 1])
            else:
                raise Exception("calculate flow mode is not allowed")
            flow_u_list.append(flow_u)
            flow_v_list.append(flow_v)
        return flow_u_list, flow_v_list

    def _turn_imgs_to_mff_input(self, rgb_list, flow_u_list, flow_v_list, indices):
        """**将连续n帧图片数据转化为对应采样与数据融合的输入数据**

        Args:
            rgb_list: rgb图片，PIL.Image.Image类型
            flow_u_list: flow_u图片，PIL.Image.Image类型
            flow_v_list: flow_v图片，PIL.Image.Image类型
            indices: 采样索引
        """
        indices = indices - 1
        num_frames = len(rgb_list)
        images = []
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                if self.modality == 'RGBFlow':
                    if i == self.new_length - 1:
                        seg_imgs = [Image.fromarray(rgb_list[p]).convert('RGB')]
                    else:
                        if p == num_frames:
                            seg_imgs = [Image.fromarray(flow_u_list[p - 1]).convert('L'),
                                        Image.fromarray(flow_v_list[p - 1]).convert('L')]
                        else:
                            seg_imgs = [Image.fromarray(flow_u_list[p]).convert('L'),
                                        Image.fromarray(flow_v_list[p]).convert('L')]
                else:
                    raise Exception("self.modality {} is not allowed".format(self.modality))

                images.extend(seg_imgs)
                if p < num_frames:
                    p += 1

        # images 一组图片共（self.new_length - 1）*2 + 1张，前self.new_length - 1每次对应横纵两张光流图，最后一张对应rgb；共self.num_segments组
        process_data = self.transform(images)
        return process_data

    @classmethod
    def calculate_brox_flow_with_dll(cls, pic_gray_1, pic_gray_2):
        """调用c++ dll计算brox光流

        Args:
            pic_gray_1: 第一张图片（灰度图）
            pic_gray_2: 第二张图片（灰度图）
        """
        start_time = time.time()

        # 转化np为uchar*
        frame_data1 = np.asarray(pic_gray_1, dtype=np.uint8)
        frame_data1 = frame_data1.ctypes.data_as(ctypes.c_char_p)
        frame_data2 = np.asarray(pic_gray_2, dtype=np.uint8)
        frame_data2 = frame_data2.ctypes.data_as(ctypes.c_char_p)

        p = cls.lib.calulateFlow_kkk(pic_gray_1.shape[0], pic_gray_1.shape[1], frame_data1,
                                     pic_gray_2.shape[0], pic_gray_2.shape[1], frame_data2)

        # 转化uchar*为np
        ret = np.array(np.fromiter(p, dtype=np.uint8, count=pic_gray_1.shape[0] * pic_gray_1.shape[1]
                                                            + pic_gray_2.shape[0] * pic_gray_2.shape[1]))
        ret_u = ret[:pic_gray_1.shape[0] * pic_gray_1.shape[1]].reshape(pic_gray_1.shape)
        ret_v = ret[pic_gray_1.shape[0] * pic_gray_1.shape[1]:].reshape(pic_gray_2.shape)

        cls.lib.release_kkk(p)

        print("calculate flow use time: {}s".format(time.time() - start_time))

        return ret_u, ret_v

    @classmethod
    def calculate_deep_flow(cls, frame1, frame2):
        start_time = time.time()

        flow = cls.inst.calc(frame1, frame2, None)

        print("calculate flow use time: {}s".format(time.time() - start_time))

        return flow[:, :, 0], flow[:, :, 1]
