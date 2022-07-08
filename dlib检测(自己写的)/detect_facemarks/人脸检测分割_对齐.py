# encoding:utf-8

import dlib
import cv2
import numpy as np
import math
import ACE图像增强


def rect_to_bb(rect):  # 获得人脸矩形的坐标信息，矩形：左边框x坐标，顶边框y坐标，宽w，高h
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def face_alignment(faces):
    # 获取人脸特征点检测器，检测人脸的特征点
    predictor = dlib.shape_predictor("./data/data_dlib/shape_predictor_68_face_landmarks.dat")  # 用来预测关键点
    faces_aligned = []  # 空列表用于存储对齐完成的图像组
    # 遍历人脸框图像组
    for face in faces:
        # 人脸特征点标注
        # rectangle用来画矩形，左上点（x1,y1），右下点(x2,y2)，彩图：shape[0]指高，shape[1]指宽，shape[2]指通道数
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        # 输入的灰度或RGB图像，rect开始内部人脸检测的边界框的位置信息
        # 返回68个特征点的位置shape是像素坐标的列表，是<_dlib_pybind11.full_object_detection>类型
        # dlib::full_object_detection会包含一个成员rect矩阵，和另外一个成员 parts点
        shape = predictor(np.uint8(face), rec)  # 注意输入的必须是uint8类型的array
        # 37~60是嘴和眼的关键点序号，注意关键点的顺序，这个在网上可以找,shape.part(i).x或y就是第i个特征点的x/y
        for j in range(36, 60):
            x = shape.part(j).x
            y = shape.part(j).y
            pos = (x, y)
            cv2.circle(face, pos, 1, (255, 0, 0), 2)  # cv2.circle画圆 （img,(x,y),r,color,小数点位数）
            # 获取在局部框图片中，第一个和第二个点的坐标（相对于图片而不是框出来的人脸）
            print("第{}个点".format(j),pos)

        # 仿射对齐变换（旋转）
        eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2,  # 计算两眼的中心坐标
                      (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)  # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
        faces_aligned.append(RotImg)
    return faces_aligned


def my_face_recognition_alignment(img_path):  # 原图标框对齐，新图像组是框内的图像
    im_raw = cv2.imread(img_path).astype('uint8')

    # 进行gamma变换图像增强
    # im_raw = adjust_gamma(im_raw, 2)

    # 获取人脸检测器detector，检测有多少个人脸框
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    # rects为detector（）返回值，是所有人脸框的矩形，坐标为[(x1, y1)，(x2, y2)] ，左上（x1,y1），右下（x2,y2）
    rects = detector(gray, 1)
    src_faces = []

    # enumerate(可遍历序列)：将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，新的索引序列每个元素为（数据下标，数据），如['a','b','c']变为[(0,'a'),(1,'b'),(2,'c')]
    for (i, rect) in enumerate(rects):
        # i是第i个人脸，rect是第i个人脸框矩形左上（x1,y1），右下（x2,y2）
        # 返回矩形：左边框x坐标，顶边框y坐标，宽w，高h
        (x, y, w, h) = rect_to_bb(rect)
        # 截取人脸框内的图像detect_face
        detect_face = im_raw[y:y + h, x:x + w]
        # 在空图上添加人脸框内图像，形成人脸框内的图像组src_faces
        src_faces.append(detect_face)
        #  在原图是绘制矩形
        cv2.rectangle(im_raw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #  在原图上标号从1开始，给每个人脸标号
        cv2.putText(im_raw, "Person face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # 人脸对齐，仅传入人脸框内的图像组src_faces，返回对齐后的图像组，每个框对齐返回一张图
    faces_aligned = face_alignment(src_faces)
    cv2.imshow("src", im_raw)
    i = 0
    # 遍历显示返回的对齐的图像组
    for face in faces_aligned:
        cv2.imshow("det_{}".format(i), face)
        i = i + 1
    print("检测到",i,"个人脸")
    cv2.waitKey(0)


# gamma变换图像增强，Gamma变换就是用来图像增强，其提升了暗部细节，通过非线性变换
# 让图像从暴光强度的线性响应变得更接近人眼感受的响应，即将漂白（相机曝光）或过暗（曝光不足）的图片，进行矫正，灰度图的均衡化
def adjust_gamma(image, gamma=1.0):
    invgamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invgamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)  # 查找表与图片的元素一一对应


if __name__ == "__main__":
    # 不要滤波，滤波会导致图像锐度不够，难以识别
    # 先进行图像增强（自动色彩均衡（ACE）快速算法/gamma增强），再识别人脸，然后截取框内图像，形成人脸图像组，然后传给对齐函数，进行框内人脸对齐，并标注
    # my_face_recognition_alignment("./data/data_faces/6.jpg")
    my_face_recognition_alignment("ACE_test.jpg")
    # 这个是已经经过ACE增强后的图像
    # ACE算法源自retinex算法，可以调整图像的对比度，实现人眼色彩恒常性和亮度恒常性，该算法考虑了图像中颜色和亮度的空间位置关系，进行局部特性的自适应滤波，
    # 实现具有局部和非线性特征的图像亮度与色彩调整和对比度调整，同时满足灰色世界理论假设和白色斑点假设。