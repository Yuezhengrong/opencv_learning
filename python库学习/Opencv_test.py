import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#一个图像由行列的像素组成，一个像素又由B,G,R三通道的值构成
#waitKey(0)将无限地显示窗口，直到任何按键按下(它适合于图像显示)。
#waitKey(25)将显示一个框架。25毫秒后，显示将自动关闭。(如果你把它放到一个循环中去读。视频，它将显示视频帧逐帧。
#uint8是专门用于处理图像的数据类型，代表0-255的整数

def read_demo():#读入一张图片
    image=cv.imread("C:/Users/Lenovo/Desktop/OIP-C.jpg")#读入图片

    #cv.namedWindow("input",cv.WINDOW_AUTOSIZE)
    cv.imshow("input",image)#创建窗口
    cv.waitKey(0)#保持窗口
    cv.destroyWindow()#结束时可以关闭窗口

def color_space_demo():#颜色空间
    image = cv.imread("C:/Users/Lenovo/Desktop/OIP-C.jpg")#RRG,0~255,三通道设备无关的颜色空间
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 颜色空间转化BGR to GRAY
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  ##颜色空间转化BGR to HSV （H 0~100，S/V 0~255）
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)  # 颜色空间转化BGR to LAB
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)  # 颜色空间转化BGR to YUV
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", image)
    cv.imshow("gray", gray)
    cv.imshow("hsv", hsv)
    cv.imshow("lab", lab)
    cv.imshow("yuv", yuv)

    mask=cv.inRange(hsv,(35,43,46),(77,255,255))#绿幕抠图
    cv.bitwise_not(mask,mask)#
    result=cv.bitwise_and(image,image,mask=mask)
    cv.imshow("mask",mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

def mat_demo():#图像创建复制
    image = cv.imread("C:/Users/Lenovo/Desktop/OIP-C.jpg")  # RRG,0~255,三通道设备无关的颜色空间
    print(image.shape)#H高,W宽,C通道数(深度)，所有图像就是numpy数组
    print(image)
    #灰度是有一个通道
    roi=image[10:50,10:50,:]
    blank=np.zeros_like(image)
    #blank[10:200,10:200,:]=image[10:200,10:200,:]#宽从10到200，高从10到200，拷贝复制，：代表通道数，注意灰度图像无此项
    blank=np.copy(image)#拷贝的是副本
    #blank=image#两个是同一对象
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", image)
    #cv.imshow("blank", roi)
    cv.imshow("blank", blank)
    cv.waitKey(0)
    cv.destroyAllWindows()

def piexl_demo():#图像像素值修改
    image = cv.imread("C:/Users/Lenovo/Desktop/OIP-C.jpg")  # RRG,0~255,三通道设备无关的颜色空间
    print(image.shape)#H高,W宽,C通道数(深度)，所有图像就是numpy数组
    h,w,c=image.shape
    for row in range(h):
        for col in range(w):
            b,g,r=image[row,col]
            image[row,col]=(255-b,255-g,255-r)#通过遍历图片行列的每个像素值，对每个像素的b，g，r取反
    cv.imwrite("C:/Users/Lenovo/Desktop/OIP-D.jpg",image)#保存图片对象到本地
    cv.imshow("input", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def math_demo():#图像像素值的算术操作+-*/
    image = cv.imread("C:/Users/Lenovo/Desktop/OIP-C.jpg")  # RRG,0~255,三通道设备无关的颜色空间
    print(image.shape)#H高,W宽,C通道数(深度)，所有图像就是numpy数组
    h,w,c=image.shape
    blank=np.zeros_like(image)
    blank[:,:]=(50,50,50)#每个像素的BGR都赋值50
    add=cv.add(image,blank)#图片加法：必须大小/通道数相等
    sub=cv.subtract(image,blank)#图片减法：必须大小/通道数相等
    #div=cv.divide(image,blank)#图片除法：必须大小/通道数相等，每个像素BGR的值对应相除
    mul = cv.multiply(image, blank)  # 图片除法：必须大小/通道数相等，每个像素BGR的值对应相除
    #cv.imshow("blank",blank)
    #cv.imshow("add", add)
    #cv.imshow("sub", sub)
    #cv.imshow("div", div)
    cv.imshow("mul", mul)
    cv.imshow("input", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def nothing(x):
    print(x)
def Trackbar_demo():#拖动条调整亮度
    image = cv.imread("C:/Users/Lenovo/Desktop/OIP-C.jpg")  # RRG,0~255,三通道设备无关的颜色空间
    cv.namedWindow("input",cv.WINDOW_AUTOSIZE)#创建窗口
    cv.imshow("input", image)
    cv2.createTrackbar("lightness", "input", 0, 100, nothing)  # 创建名为lightness的拖动条,放在input的窗口内，回调函数设为nothing
    blank=np.zeros_like(image)
    while True:
        pos=cv.getTrackbarPos("lightness", "input")#pos获得拖动条的值
        blank[:, :] = (pos, pos, pos)  # 每个像素的BGR都赋值为拖动条的值，实时改变图像色彩
        result=cv.add(image,blank)
        #cv.imshow("blank",blank)
        cv.imshow("result", result)
        c = cv.waitKey(1)  # c接受一个返回值
        if (c == 27):#在窗口输ESC，才满足条件
            break
    cv.destroyAllWindows()

def keys_demo():#利用键盘输入函数wait_key()
    image = cv.imread("C:/Users/Lenovo/Desktop/OIP-C.jpg")  # RRG,0~255,三通道设备无关的颜色空间
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)  # 创建窗口
    cv.imshow("input", image)
    while True:
        c = cv.waitKey(1)  # c接受一个键盘返回值
        gray_hsv = image
        if c==49:#输入2
            gray_hsv=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
            cv.imshow("result", gray_hsv)
        if c==50:#输入1
            gray_hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
            cv.imshow("result", gray_hsv)
        if (c == 27):  # 在窗口输入ESC，才满足条件
            break
    cv.destroyAllWindows()

def color_table_demo():#使用opencv色彩表
    colormap = [#创建色彩表
        cv.COLORMAP_AUTUMN,
        cv.COLORMAP_BONE,
        cv.COLORMAP_JET,
        cv.COLORMAP_WINTER,
        cv.COLORMAP_RAINBOW,
        cv.COLORMAP_OCEAN,
        cv.COLORMAP_SUMMER,
        cv.COLORMAP_SPRING,
        cv.COLORMAP_COOL,
        cv.COLORMAP_PINK,
        cv.COLORMAP_HOT,
        cv.COLORMAP_PARULA,
        cv.COLORMAP_MAGMA,
        cv.COLORMAP_INFERNO,
        cv.COLORMAP_PLASMA,
        cv.COLORMAP_VIRIDIS,
        cv.COLORMAP_CIVIDIS,
        cv.COLORMAP_TWILIGHT,
        cv.COLORMAP_TWILIGHT_SHIFTED
    ]
    image = cv.imread("C:/Users/Lenovo/Desktop/OIP-C.jpg")  # RRG,0~255,三通道设备无关的颜色空间
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)  # 创建窗口
    cv.imshow("input", image)
    index=0
    while True:
        c = cv.waitKey(1)  # 等待1ms，c接受一个键盘返回值
        dst=cv.applyColorMap(image,colormap[index%19])#索引按照19一循环，按照所索引创建色彩表的颜色对象dst
        index+=1
        cv.imshow("color style",dst)#显示色彩表的颜色对象dst
        if(c==27):#ESC退出
            break
    cv.destroyAllWindows()

def bitwise_demo():#位运算
    b1=np.zeros((400,300,3),dtype=np.uint8)
    b1[:,:]=(0,255,255)
    b2=np.zeros((400,300,3),dtype=np.uint8)
    b2[:,:]=(127,255,0)
    dst1=cv.bitwise_and(b1,b2)
    dst2=cv.bitwise_or(b1,b2)
    cv.imshow("b1", b1)
    cv.imshow("b2", b2)
    cv.imshow("bitwise_and", dst1)
    cv.imshow("bitwise_or",dst2)
    cv.waitKey(0)
    cv.destroyAllWindows()

def channels_split_demo():#通道分类与合并
    image=cv.imread("C:/Users/Lenovo/Desktop/OIP-C.jpg")
    cv.imshow("image1", image[:,:,0])# : 表示像素取从0到结束,最后一个下标表示第几个通道
    cv.imshow("image2", image[:, :, 1])
    cv.imshow("image3", image[:, :, 2])
    cv.imshow("image", image)
    mv=cv.split(image)#mv为包含三个一维数组的列表，每个元素为一个一维数组，即一个通道的所有值
    print(mv)
    result=cv.merge(mv)
    cv.imshow("result",result)
    cv.waitKey(0)
    cv.destroyAllWindows()

def pixel_stat_demo():#求图像三通道的均值和方差
    image=cv.imread("C:/Users/Lenovo/Desktop/OIP-C.jpg")
    cv.imshow("image", image)
    means,dev=cv.meanStdDev(image)#求均值means，方差dev
    print(np.max(image[:,:,0]))#第一个通道的最大值
    print(np.max(image[:, :, 1]))#第二个通道的最大值
    print(np.max(image[:, :, 2]))#第三个通道的最大值
    # dev=0 说明是净纸
    print(means,dev)
    cv.waitKey(0)
    cv.destroyAllWindows()

def draw_demo():#画几何图形（主要是长方形框，文本）
    b1=cv.imread("C:/Users/Lenovo/Desktop/OIP-C (1).png")
    cv.rectangle(b1,(70,30),(180,150),(0,0,255),2,8,0)#绘制长方形(在哪个图片里，(左上角)，（右下角），（颜色三通道），线宽px，填充，填充)
    #cv.circle(b1,(200,200),100,(255,0,0),2,8,0)#绘制圆形（在哪个图片里，（圆心），，（颜色通道），线宽px，填充，填充）
    cv.putText(b1,"99% face",(50,50),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2,8)#绘制文本（在哪个图片里，文本，起点，字体，字号,颜色通道，粗细，填充）
    cv.imshow("b1", b1)
    cv.waitKey(0)
    cv.destroyAllWindows()

def norm_demo():#图像像素类型转换与归一化
    image=cv.imread("C:/Users/Lenovo/Desktop/OIP-C (1).png")
    #cv.namedWindow("norm_dome",cv.WINDOW_AUTOSIZE)
    print(image/255.0)#像素值是0-255之间的数
    result =np.zeros_like(np.float32(image))#创建image大小的float图片
    cv.normalize(np.float32(image),result,0,1,cv.NORM_MINMAX,dtype=cv.CV_32F)#像素值转换为uint型，0-255
    #像素转换为uint型的方法：1。result=np.uint8(image)不好 2。cv.normalize(np.float32(image),result,0,1,cv.NORM_MINMAX,dtype=cv.CV_32F) 好
    cv.imshow("norm_demo",result)#imshw可以显示0-1的浮点数和0-255的整数
    cv.waitKey(0)
    cv.destroyAllWindows()

def resize_demo():#图片的放缩与插值（图像重采样）
    image = cv.imread("C:/Users/Lenovo/Desktop/OIP-C (1).png")
    h,w,c=image.shape
    dst=cv.resize(image,(0,0),fx=2,fy=2,interpolation=cv.INTER_CUBIC)#放缩dst为image的fx和fy为宽高放大倍数,插值方式=双立方插值
    cv.imshow("resize_demo", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

def flip_demo():#图像翻转
    image = cv.imread("C:/Users/Lenovo/Desktop/OIP-C (1).png")
    result1=cv.flip(image,0)#图像翻转函数，0上下翻转，1左右对称翻转
    result2 = cv.flip(image, 1)  # 图像翻转函数，0上下翻转，1左右对称翻转，-1上下+左右翻转
    cv.imshow("image_demo", image)
    cv.imshow("flip1_demo", result1)
    cv.imshow("flip2_demo", result2)
    cv.waitKey(0)
    cv.destroyAllWindows()

def rotate_demo():#图像旋转
    image = cv.imread("C:/Users/Lenovo/Desktop/OIP-C (1).png")
    h, w, c =image.shape
    M=np.zeros((2,3),dtype=np.float32)#定义旋转矩阵M,2*3大小
    #定义选址角度
    alpha=np.cos(np.pi/4.0)#旋转Π/4的角度，45度
    beta=np.sin(np.pi/4.0)#旋转Π/4的角度，45度
    #根据角度初始化旋转矩阵M
    M[0,0]=alpha
    M[1, 1] =alpha
    M[0, 1] =beta
    M[1, 0] =-beta
    cx=w/2
    cy=h/2
    tx=(1-alpha)*cx-beta*cy
    ty=beta*cx+(1-alpha)*cy
    M[0,2]=tx
    M[1,2]=ty
    new_w=int(h*np.abs(beta)+w*np.abs(alpha))#计算新的宽
    new_h=int(h*np.abs(alpha)+w*np.abs(beta))#计算新的高
    M[0, 2] += new_w/2-cx
    M[1, 2] += new_h/2-cy
    # 执行旋转
    dst=cv.warpAffine(image,M,(new_w,new_h))#（原图，旋转矩阵M，（新的宽，新的高））
    cv.imshow("rotate_demo", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

def video_demo():#视频/摄像头读取
    cap=cv.VideoCapture(0)#创造视频捕捉对象cap，（0）：是读摄像头；（路径）：是读视频文件
    w=cap.get(cv.CAP_PROP_FRAME_WIDTH)#获取视频文件的宽度
    h=cap.get(cv.CAP_PROP_FRAME_HEIGHT)#获取视频文件的高度
    fps=cap.get(cv.CAP_PROP_FPS)#获取视频分辨率(帧率：每秒播放多少张)
    print(w,h,fps)
    while True:#循环加载显示视频
        ret,frame=cap.read()#如果读到视频，返回ret=True，frame=读到的视频对象的每一帧
        frame=cv.flip(frame,1)#因为摄像头读到的是反的，翻转回来；但视频文件不用翻转！！
        if ret is True:
            cv.imshow("frame",frame)
            c=cv.waitKey(10)#每10ms加载一次
            if c==27:
                break
    cv.destroyAllWindows()


def keep_video():#保存视频
    cap = cv.VideoCapture("C:/Users/Lenovo/Desktop/video.avi")  # 创造视频捕捉对象cap，0：读摄像头；路径：读视频文件
    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)  # 获取视频文件的宽度
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # 获取视频文件的高度
    fps = cap.get(cv.CAP_PROP_FPS)  # 获取视频分辨率(帧率：每秒播放多少张)
    out=cv.VideoWriter("C:/Users/Lenovo/Desktop/lane.avi",cv.CAP_ANY,int(cap.get(cv.CAP_PROP_FOURCC)),fps,(int(w),int(h)),True)#视频文件保存器(保存地址,编码格式,帧率，宽高，是否为彩色)
    #具体编码方式自己查,cap.get(cv.CAP_PROP_FOURCC)，是获取原视频的编码
    print(w, h, fps)
    while True:  # 循环加载显示视频
        ret, frame = cap.read()  # 如果读到视频，返回ret=True，frame=读到的视频对象的每一帧
        if ret is not True:
            break
        hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)#可将视频转换颜色空间
        cv.imshow("frame", frame)
        out.write(hsv)#保存器 保存视频文件
        c = cv.waitKey(10)  # 每10ms加载一次
        if c == 27:
            break
    cv.destroyAllWindows()

def image_hist():#图像直方图：每个通道 0-255像素值 出现的次数
    image=cv.imread("C:/Users/Lenovo/Desktop/OIP-C (1).png")
    cv.imshow("input",image)
    color=("blue","green","red")
    for i,color in enumerate(color):#返回 enumerate 枚举序列的对象
        hist=cv.calcHist([image],[i],None,[256],[0,256])#calcHist计算每个颜色通道下，不同像素值的频率直方图的数据hist
        print(hist)
        plt.plot(hist,color=color)#matplotlib矩阵可视化
        plt.xlim([0,255])
    plt.show()
    cv.waitKey(10)
    cv.destroyAllWindows()

def conv_demo():#图片卷积
    image = cv.imread("C:/Users/Lenovo/Desktop/OIP-C (1).png")
    cv.imshow("input", image)
    mohu=cv.blur(image,(5,5))#模糊卷积：(x,x)卷积核大小为x*x
    gauss=cv.GaussianBlur(image,(5,5),15)#高斯卷积模糊：(x,x)卷积核大小为x*x，且x一定为奇数
    gauss_double=cv.bilateralFilter(image,0,100,1)#高斯双边模糊，磨皮
    cv.imshow("mohu",mohu)
    cv.imshow("gouss",gauss)
    cv.imshow("gauss_double",gauss_double)
    cv.waitKey(0)
    cv.destroyAllWindows()


model_bin = "C:/Users/Lenovo/Desktop/opencv_face_detector_uint8.pb"  # 这个路径改一下
config_text = "C:/Users/Lenovo/Desktop/opencv_face_detector.pbtxt"  #这个路径改一下
def face_detection_demo():#人脸识别
    net = cv.dnn.readNetFromTensorflow(model=model_bin, config=config_text)#创建网络
    cap = cv.VideoCapture(0)#视频获取对象
    while True:
        ret,frame = cap.read()#读取 视频对象和视频格式
        h, w, c = frame.shape
        if ret is not True:
            break
        # NCHW
        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blob)
        outs = net.forward() # 1x1xNx7
        for detection in outs[0, 0, :, :]:
            score = float(detection[2])
            if score > 0.5:
                left = detection[3] * w
                top = detection[4] * h
                right = detection[5] * w
                bottom = detection[6] * h
                cv.rectangle(frame, (np.int(left), np.int(top)), (np.int(right), np.int(bottom)), (0, 0, 255), 2, 8, 0)
        cv.imshow("my_face_detector", frame)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()
    cap.release()



if __name__ == "__main__":
    face_detection_demo()