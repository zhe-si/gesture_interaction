# This code hase been acquired from TRN-pytorch repository
# 'https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py'
# which is prepared by Bolei Zhou
#
# Processing the raw dataset of Jester
#
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Created by Bolei Zhou, Dec.2 2017

# 用于生成数据集的索引文件'%s-labels.csv'(标签种类), 'val_videofolder.txt'(验证集数据对应标签), 'train_videofolder.txt'(训练集数据对应标签)
# 用于Jester, 其他数据集有另外的处理脚本

import os
import pdb

dataset_name = 'jester-v1'

# 要求有所有标签名, 会将其排序后存到category.txt
with open('%s-labels.csv' % dataset_name) as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()  # 删除末尾的空字符
    categories.append(line)
categories = sorted(categories)
with open('category.txt', 'w') as f:
    f.write('\n'.join(categories))

dict_categories = {}  # 产生分类的标签名->标签序号的字典(序号为根据排列好的标签顺序生成的)
for i, category in enumerate(categories):
    dict_categories[category] = i

# jester验证集和训练集对应的数据号（文件夹名）-> 标签名。格式为：34870;Drumming Fingers
files_input = ['%s-validation.csv' % dataset_name, '%s-train.csv' % dataset_name]
files_output = ['val_videofolder.txt', 'train_videofolder.txt']
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = f.readlines()
    folders = []  # 保存每个标签的文件夹编号
    idx_categories = []  # 保存该文件对应标签的序号
    for line in lines:
        line = line.rstrip()
        items = line.split(';')
        folders.append(items[0])
        # idx_categories.append(os.path.join(dict_categories[items[1]]))  # TODO: 此处有问题，os.path.join(int??), 要干什么用？
        idx_categories.append(dict_categories[items[1]])
    output = []  # 字符串列表，每个字符串为"数据文件夹编号 文件夹中图片数量 标签序号"
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        # counting the number of frames in each video folders
        dir_files = os.listdir(os.path.join('20bn-%s' % dataset_name, curFolder))  # 得到某个目录下所有文件名（不是完整路径）
        output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
        print('%d/%d' % (i, len(folders)))
    with open(filename_output, 'w') as f:
        f.write('\n'.join(output))
