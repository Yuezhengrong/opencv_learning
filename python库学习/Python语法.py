# 类与对象
class Car:
    name = "s"

    def __init__(self, name):  # 构造函数，self是调用函数的当前对象
        self.name = name

    def print(self):
        print(self.name)


# 继承
class GM(Car):
    pirce = 100

    def __init__(self, name, pirce):
        super().__init__(name)
        self.pirce = pirce

    # 重写父类方法
    def print(self):
        print(self.name, self.pirce)


# 数据存储与读取csv和json文件
# csv里全是字符串类型的数据，除了下列操作，还可以用pands更加简洁的实现
# 读csv格式：用，将字符数据分隔开，csv可以用excel打开
with open("olist_sellers_dataset.csv", "r", encoding="gbk") as f:  # 打开文件f
    ls = []  # 使用列表对文件内容进行储存
    for line in f:  # 对每行进行读取
        ls.append(line.strip("\n").split(","))  # 去掉换行符\n，用，分割元素，将每行分割的结果添加到ls
for res in ls:
    print(res)
# 写csv格式：用，将字符数据分隔开，可以用excel打开
ls = [['编号', '数学成绩', '语文成绩'],
      ['1', '100', '100'],
      ['2', '99', '90']]
with open("olist_sellers_dataset.csv", "w", encoding="gbk") as f:
    for row in ls:
        f.write(",".join(row) + "\n")  # 用逗号join组合成字符串形式，加速换行符
# json里全是字典类型的数据
import json
# 写入json文件
scores = {
    "yzr": {"math": 99, "english": 98},
    "zrz": {"math": 99, "english": 98}
}
with open("score.json",'w',encoding="utf-8") as f:
    json.dump(scores,f,indent=4,ensure_ascii=False)  # dump写入，ident表示字符串换行缩进，ensure_ascii表示显示中文
# 读出json文件
with open("score.json",'r',encoding="utf-8") as f:
    scores2=json.load(f)    # 加载整个文件
    for k,v in scores2.items():
        print(k,v)


if __name__ == "__main__":
    # 浅拷贝-指向同一存储空间,别名
    list1=[1,2,3]
    list2=list1
    list2[2]=4
    print(list1)  # [1,2,4]
    # 深拷贝-内容复制
    list3=list1.copy()
    list3[2] = 3
    print(list1)  # [1,2,4]



