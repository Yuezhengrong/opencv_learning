import numpy
import numpy as np

# numpy的array类
print(numpy.array([1, 2, 3]).reshape(3, 1))

# numpy.arange(start, stop, step, dtype)
print(numpy.arange(12).reshape(3, 4))

# numpy的API linspace(start, stop, num, endpoint, retstep, dtype)
print(numpy.linspace(0, 10, 4).reshape(2, 2))

# logscale(start, stop, num, endpoint, base, dtype)
print(numpy.logspace(0, 10, 4).reshape(2, 2))

# ones(shape, dtype = None, order = 'C')
print(numpy.ones([3, 3]))

# numpy.zeros(shape, dtype = float, order = 'C')
print(np.zeros((2, 2)))

# 随机数
print(numpy.random.rand(2, 2))  # 0-1分布
print(numpy.random.uniform(5, 10, (2, 2)))  # 指定上下界
print(numpy.random.randint(10, 20, (2, 2)))  # Int
rng = numpy.random.default_rng()  # 用生成器
print(rng.integers(10, 20, (2, 2)))
print(rng.normal(10, 10, (2, 2)))  # 高斯分布

# 文件读取
numpy.save("../VOCdevkit/VOC2007/a.npy", numpy.random.rand(5, 5))  # 存入磁盘文件
b = numpy.load("../VOCdevkit/VOC2007/a.npy")  # 加载磁盘文件
print(b)

# 数组的属性
arr = numpy.array(rng.integers(10, 30, (4, 5)))
print(arr.shape)#形状
print(arr.dtype)#数据类型
print(arr.size)#大小
print(arr.base)

print(numpy.sum(arr))  # 求和
print(arr.max())
print(arr.min(axis=1))  # axis就是数组维级dimension,列表或元组的括号深度，shape 得到的形状实际上是数组在每个轴 Axis 上面的元素数量，而 .shape 的长度的表明了数组的维度
print("2维数组，axis=0，在每一列中寻找min，axis=1，在每一行中寻找min")
# 2维数组，axis=0，在每一列中寻找min，axis=1，在每一行中寻找min
print(numpy.median(arr))  # 中位数
print(numpy.average(arr))  # 求平均值

print(numpy.std(arr, axis=0))  # 标准差
# print(numpy.expand_dims(arr,axis=(1,3)))
a = numpy.array([[1, 2], [3, 4]])
b = numpy.array([[4, 3], [2, 1]])
print(a)
print(b)
print(a * b)  # 对应位置相乘
print(numpy.dot(a, b))  # 矩阵相乘
print(a.dot(b))
