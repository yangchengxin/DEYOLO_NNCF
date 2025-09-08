import xml.etree.ElementTree as ET
import os, cv2
import numpy as np
from os import listdir
from os.path import join
from tqdm import tqdm

classes = []


# 这个函数接收图像尺寸和一个边界框坐标，然后将边界框的坐标转换为相对于图像尺寸的比例值。
# 这是YOLO模型所需要的格式，其中(x, y)是边界框中心点的位置，(w, h)是边界框的宽度和高度，所有值都在0到1之间。
def convert(size, box):
    dw = 1. / (size[0])  # 图像高度的倒数
    dh = 1. / (size[1])  # 图像高度的倒数
    x = (box[0] + box[1]) / 2.0 - 1  # 计算边界框中心点的 x 坐标
    y = (box[2] + box[3]) / 2.0 - 1  # 计算边界框中心点的 y 坐标
    w = box[1] - box[0]  # 边界框的宽度
    h = box[3] - box[2]  # 边界框的高度
    # 归一化
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(xmlpath, xmlname):
    # 打开XML文件以读取内容
    with open(xmlpath, "r", encoding='utf-8') as in_file:
        # 将XML文件名转换为对应的TXT文件名
        txtname = xmlname[:-4] + '.txt'
        # 构建TXT文件的完整路径
        txtfile = os.path.join(txtpath, txtname)

        # 解析XML文件
        tree = ET.parse(in_file)
        root = tree.getroot()  # 获取XML文档的根节点

        # 从XML中获取图像文件名
        filename = root.find('filename')

        # np.fromfile 方法从文件系统中读取图像数据，返回一个 NumPy 数组
        # 使用 OpenCV 的 imdecode 函数将图像数据解码为图像，cv2.IMREAD_COLOR 表示以彩色模式加载图像
        img = cv2.imdecode(np.fromfile('{}/{}.{}'.format(imgpath, xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

        res = []  # 初始化结果列表

        # 遍历XML中的每个对象
        for obj in root.iter('object'):
            cls = obj.find('name').text  # 获取对象的类别名称
            # 如果类别还未记录，则添加到全局类别列表中
            if cls not in classes:
                classes.append(cls)
            # 获取类别的索引
            cls_id = classes.index(cls)
            # 获取边界框信息
            xmlbox = obj.find('bndbox')
            # 提取边界框的坐标
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            # 调用convert函数，将边界框坐标转换为YOLO格式
            bb = convert((w, h), b)
            # 将类别ID和边界框信息组合成字符串，并添加到结果列表中
            res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))

        # 如果有对象被检测到，则将结果写入TXT文件
        if len(res) != 0:
            with open(txtfile, 'w+') as f:
                f.write('\n'.join(res))


if __name__ == "__main__":
    postfix = 'bmp'  # 设置图像文件的扩展名
    imgpath = r'C:\Users\guest_7\Desktop\fsdownload\data\04_color\04_datasets\image'  # 设置图像文件路径
    xmlpath = r'C:\Users\guest_7\Desktop\fsdownload\data\04_color\04_datasets\label'  # 设置 XML 标注文件路径
    txtpath = r'C:\Users\guest_7\Desktop\fsdownload\data\04_color\04_datasets\txt1'  # 设置输出的 TXT 文件路径

    # 检查输出路径是否存在，如果不存在则创建
    if not os.path.exists(txtpath):
        os.makedirs(txtpath, exist_ok=True)

    list = os.listdir(xmlpath)  # 获取 XML 文件列表
    error_file_list = []  # 初始化错误文件列表

    # 遍历 XML 文件列表
    for i in tqdm(range(0, len(list))):
        try:
            # 构建 XML 文件的完整路径
            path = os.path.join(xmlpath, list[i])
            # 检查文件是否为 XML 格式
            if ('.xml' in path) or ('.XML' in path):
                convert_annotation(path, list[i])
                print(f'file {list[i]} convert success.')
            else:
                print(f'file {list[i]} is not xml format.')
        except Exception as e:
            # 捕获并处理任何异常
            print(f'file {list[i]} convert error.')
            print(f'error message:\n{e}')
            # 将发生错误的文件添加到错误文件列表中
            error_file_list.append(list[i])

    print(f'this file convert failure\n{error_file_list}')  # 打印转换失败的文件列表
    print(f'Dataset Classes:{classes}')  # 打印数据集中的所有类别