import torch
import numpy as np
import os
import pandas as pd


def linear(x, a, b):
    return a * x + b


def sigmoid(x, w=1):
    return 1 / (1 + np.sum(np.exactativep(-w * x)))


def tanh(x):
    return np.tanh(x)


def relu(x):
    if x < 0:
        return 0
    else:
        return x


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def tensor_creat():
    # tensor通常指n维向量，图像中常为3维，前2维分别是像素宽高，最后一维表示RGB通道值
    # numpy.array 与 tensor的转化
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print("type a=", type(a))
    # numpy->tensor
    t = torch.tensor(a)
    print("type b=", type(t))
    i = torch.from_numpy(a)
    print("type i=", type(i))
    # tensor->numpy
    n = t.numpy()
    print("type n=", type(n))

    # 常见的构造Tensor的函数
    k = torch.rand(2, 3)
    l = torch.ones(2, 3)
    m = torch.zeros(2, 3)
    n = torch.arange(0, 10, 2)
    print(k, '\n', l, '\n', m, '\n', n)


def tensor_base_opreation():
    k = torch.rand(2, 3)
    l = torch.ones(2, 3)
    # 查看tensor的维度信息（两种方式）
    print(k.shape)
    print(k.size())

    # tensor的运算(tensor的+-*/都是直接对对应位置的元素运算，而非矩阵的运算)
    # 按元素乘法，又叫哈德马积 Hadamard积（数学符号为 点圈）
    o = torch.add(k, l)
    print(o)
    o = k + l
    print("o=", o)

    # 真正的矩阵运算
    a = torch.arange(5).reshape(1, 5)
    b = torch.arange(5).reshap(5, 1)
    print("矩阵乘法a*b=", torch.dot(a, b))

    # tensor的索引与切片方式与numpy类似
    # 下标从0开始,( : 代表所有行/列),(a:b 代表从a到b行/列)
    # ( -1代表最后一行/列)
    print("o[:, 1]=", o[:, 1])  # 第1列所有元素
    print("o[0, :]=", o[0, :])  # 第0行所有元素
    print("o[0, 1]=", o[0, 1])  # 第0行第1列的元素
    print("o[-1, 1]=", o[-1, 1])  # 最后一行，第1列元素
    print("o[0:1, 1]=", o[0:1, 1])  # 从0到1行，第1列的所有元素

    # 对某维度求和sum,dim/axis代表某一维度
    a = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)  # a是2个3*4的数组组成的张量，三个维度值分别代表：几个、几行、几列
    print("a=", a)  # a有0、1、2两个维axis，维度的值分别是2，3，4
    # 对哪个维度求和，那个维度消失
    a_sum_axis0 = a.sum(axis=0)  # 对a的第0个维度求和
    print("a_sum_axis0=", a_sum_axis0)
    print(a_sum_axis0.shape)
    a_sum_axis1 = a.sum(axis=1)  # 对a的第1个维度求和
    print("a_sum_axis1=", a_sum_axis1)
    print(a_sum_axis1.shape)
    a_sum_axis2 = a.sum(axis=2)  # 对a的第2个维度求和
    print("a_sum_axis2=", a_sum_axis2)
    print(a_sum_axis2.shape)

    # 求均值mean、average
    a_mean = a.mean()
    print("a_mean=", a_mean)
    # 在某维度内求均值
    print("a.mean(dim=0)=", a.mean(dim=0))  # 在第0维求均值（在每个3*4矩阵内求均值）
    print("a.mean(dim=1)=", a.mean(dim=1))  # 在第1维求均值（在每行求均值）
    print("a.mean(dim=2)=", a.mean(dim=2))  # 在第2维求均值（在每列求均值）

    # 改变tensor形状的神器：view
    print(o.view((3, 2)))
    print(o.view(-1, 2))  # 自动添补值为-1的维度

    # tensor的广播机制（使用时要注意这个特性）
    # tensor形状不一致时，进行运算，会自动补齐维度，可能造成错误
    p = torch.arange(1, 3).view(1, 2)
    print(p)
    q = torch.arange(1, 4).view(3, 1)
    print(q)
    print(p + q)  # 广播

    # 扩展&压缩tensor维度的值为1的维度：squeeze，
    print(o)
    r = o.unsqueeze(1)  # 在第1维扩展（维度从第0维开始）r的维度从[2,3]->[2,1,3]
    print(r)
    print(r.shape)
    # 只有当前维度的值为1才可以压缩
    s = r.squeeze(0)  # 在第0维压缩，但第0维不是1，压缩失败
    print(s)
    print(s.shape)

    # 求和、求均值时保持维度不变，使用广播机制
    sum_a = a.sum(axis=1, keepdims=True)  # sum_a还是2*3*4的张量,只是第1维的维度值变成1，（2*1*4）
    print("a/sum_a=", a / sum_a)  # 广播机制，使得a的每个元素除以sum_a

    # 节省内存，避免原变量的内存不断开辟，id() 获取对象唯一标识（指针）
    Y = torch.rand(2, 3)
    before_id = id(Y)
    X = torch.rand(2, 3)
    Y = Y + X
    print(id(Y) == before_id)  # False说明Y运算后开辟了新内存
    # 为节省空间，我们更加支持原地操作（以+为例）
    print('原始id(Y):', id(Y))
    Y[:] = X + Y
    print('id(Y):', id(Y))
    Y += X
    print('id(Y):', id(Y))
    # 有时需要开辟新内存 直接赋值
    Y = X
    print('id(Y):', id(Y))


def tensor_degrade():
    # 自动求导
    x1 = torch.tensor(1.0, requires_grad=True)
    x2 = torch.tensor(2.0, requires_grad=True)
    y = x1 + 2 * x2
    print(y)

    # 查看每个变量导数大小。此时因为还没有反向传播，因此导数都不存在
    # print(x1.grad.data)
    # print(x2.grad.data)
    # print(y.grad.data)
    # 反向传播后看导数大小
    y.backward()
    print(x1.grad.data)
    print(x2.grad.data)


def data_preprocess():
    # 创建文件夹，os.path.join()是创建路径函数
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    # 创建CSV（逗号分隔值）文件 ../data/house_tiny.csv
    # (..是退回上一级目录的意思)
    data_file = os.path.join('..', 'data', 'house_tiny.csv')
    # 以wirte形式，打开文件
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    # csv文件为每行为一个字符串，字符串内部用","分割，读出后可以和其他类型自动转化，每行结束有一个"\n"
    # 调用pands的read_csv()函数读取csv文件的内容，到一个pands变量中
    data = pd.read_csv(data_file)
    print(data)

    # 处理缺失值
    # 典型的方法包括插值法和删除法，“NaN”项代表缺失值。插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值所在行
    # 通过pands.read得对象的位置索引iloc（非原地操作），我们将data分成inputs和outputs， 其中前者为data的前两列，而后者为data的最后一列。
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    # 对于inputs中的NaN，我们用均值替换“NaN”项
    inputs = inputs.fillna(inputs.mean())  # 数值直接求均值
    print(inputs)
    # 对于inputs中的类别值或离散值，我们将“NaN”和“Pave”分别视为一个类别，非0即1
    # get_dummies()函数可以自动将“Alley”列转换为两列“Alley_Pave”和“Alley_nan”
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)
    outputs = pd.get_dummies(outputs, dummy_na=True)
    print(outputs)

    # 将处理过的数据转化为tensor
    x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
    x, y


def PyImgLib():  # python图像库，效果和opencv类似,区别：PIL读取的图像是JpegImage类型，而Opencv读取的是numpy类型
    from PIL import Image  # 图像类
    from PIL import ImageFilter  # 滤波器类
    # PIL:Python Imaging Library
    # 路径用双斜杠\\
    image_path = "D:\\深度学习_炼丹炉\\Opencv_pytorch项目\\Pytorch\\hymenoptera_data\\train\\ants\\0013035.jpg"
    # 创建Image图像类变量
    img = Image.open(image_path)
    img.show()  # 显示图像
    img.save('1.jpg')  # 保存图像
    print(img.mode, img.size, img.format)
    # mode属性为图片的模式，RGB 代表彩色图像，L代表灰度图像
    # size 属性为图片的大小(宽度，长度)
    # format 属性为图片的格式，如常见的 PNG、JPEG 等
    grey_img = img.convert('L')  # 转化为灰度图
    # grey_img.show()

    # 彩色图像可以分离出 R、G、B 通道
    r, g, b = img.split()
    # r g b分别是三个通道下的二维像素矩阵
    # 可以将 R、G、B 通道按照一定的顺序再合并成彩色图像
    img_merge = Image.merge('RGB', (b, g, r))
    # img_merge.show()

    # 像素值操作
    out = img.point(lambda i: i * 1.2)  # 对每个像素值乘以 1.2
    # out.show()
    source = img.split()  # source里有三个二维矩阵，分别是R,G,B
    # i > 128 and 255，当 i <= 128 时，返回 False 即 0,；反之返回 255
    out = source[0].point(lambda i: i > 128 and 255)  # 对 R 通道矩阵进行二值化
    out.show()

    # Image类对象 和 Numpy 数组之间的转化
    array = np.array(img)
    print(array.shape)  # (321, 481, 3)
    out = Image.fromarray(array)

    # 图像滤波，在ImageFilter后选择具体的滤波函数
    img.filter(ImageFilter.BLUR).show()


def read_data():  # 数据加载
    # 抽象Dataset类,inputs和lable
    from torch.utils.data import Dataset
    from PIL import Image  # 图像类
    # 绝对路径用\\分隔，相对路径用/分隔
    class MyData(Dataset):  # 创建自己的图片数据集，继承Dataset
        def __init__(self, root_dir, label_dir):  # 构造函数，传入数据集总目录相对路径和具体某类的文件夹相对路径
            # root_dir是数据集文件夹路径，如"Pytorch/hymenoptera_data/train"
            # lable_dir是在数据集文件夹中，某个类的文件夹路径，如"ants"
            self.root_dir = root_dir  # 将root_dir设为类的全局变量
            self.label_dir = label_dir  # 将label_dir设为类的全局变量
            # join():路径拼接函数，返回该Dataset对象的路径
            self.path = os.path.join(root_dir, label_dir)
            # listdir(path)：将path文件夹下的所有文件的名称，变成一个列表对象
            self.img_path_list = os.listdir(self.path)

        def __getitem__(self, index):  # 从0开始，通过index获取对应图片的名字，再生成地址，返回 图片对象和 该图片的数据标签
            img_name = self.img_path_list[index]
            img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
            img = Image.open(img_item_path)
            return img, self.label_dir

        def __len__(self):  # 返回该类别数据集的图片数量
            return len(self.img_path_list)

    # 实例化MyData类进行测试,分别创建ants和bees的数据集对象进行访问
    ant_dataset = MyData("hymenoptera_data/train", "ants")
    img, label = ant_dataset.__getitem__(30)
    img.show()
    print(label)
    print(ant_dataset.__len__())  # 124个

    bee_dataset = MyData("hymenoptera_data/train", "bees")
    img, label = bee_dataset.__getitem__(10)
    img.show()
    print(label)
    print(bee_dataset.__len__())  # 121个

    # 创建训练集train对象,里面的图片对象是ants和bees的数据集的拼接
    train_dataset = ant_dataset + bee_dataset
    img, label = train_dataset.__getitem__(100)
    img.show()
    print(label)
    print(train_dataset.__len__())  # 245个


# 抽象Dataset类,inputs和lable
from torch.utils.data import Dataset
from PIL import Image  # 图像类


# 绝对路径用\\分隔，相对路径用/分隔
class MyData(Dataset):  # 创建自己的图片数据集，继承Dataset
    def __init__(self, root_dir, label_dir):  # 构造函数，传入数据集总目录相对路径和具体某类的文件夹相对路径
        # root_dir是数据集文件夹路径，如"Pytorch/hymenoptera_data/train"
        # lable_dir是在数据集文件夹中，某个类的文件夹路径，如"ants"
        self.root_dir = root_dir  # 将root_dir设为类的全局变量
        self.label_dir = label_dir  # 将label_dir设为类的全局变量
        # join():路径拼接函数，返回该Dataset对象的路径
        self.path = os.path.join(root_dir, label_dir)
        # listdir(path)：将path文件夹下的所有文件的名称，变成一个列表对象
        self.img_path_list = os.listdir(self.path)

    def __getitem__(self, index):  # 从0开始，通过index获取对应图片的名字，再生成地址，返回 图片对象和 该图片的数据标签
        img_name = self.img_path_list[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        return img, self.label_dir

    def __len__(self):  # 返回该类别数据集的图片数量
        return len(self.img_path_list)


def tensor_board():
    # 从tensorboard导入SummaryWriter类
    from torch.utils.tensorboard import SummaryWriter
    # 显示训练日志对象，显示loss图
    writer = SummaryWriter("logs")
    # 创建日志文件夹对象logs

    img = Image.open("hymenoptera_data/train/ants/0013035.jpg")
    img_array = np.array(img)  # 将img从Image转为numpy.array类型
    writer.add_image("test", img_array, 1, dataformats="HWC") # CHW:通道、高、宽。  HWC：高、宽、通道。

    for i in range(100):
        writer.add_scalar("y=2x", 2 * i, i)  # logs标题名称，value，step
    writer.close()
    # pycharm终端cd进入logs文件夹的上一级目录，输入： tensorboard --logdir=logs --port=6007
    # 打开logs文件

# def image_transform():  # 图片变换扩展数据集


def data_load():
    # 抽象DataLoader类
    from torch.utils.data import DataLoader
    class MyDataloador(DataLoader):
        def __init__(self):
            print(1)


if __name__ == "__main__":
    tensor_board()
