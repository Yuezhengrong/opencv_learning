import requests
from urllib.request import urlopen


def my_request_get():
    query = input("你想搜索的关键词")
    url = f'http://www.sogou.com/web?query={query}'
    dic = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"}
    response = requests.get(url, headers=dic)  # get模拟用户发送请求，headers是"User-Agent":的字典
    print(response)  # 输出查看服务器是否成功响应
    print(response.text)  # 拿到源代码
    response.close()

def my_request_post():
    url = 'http://fanyi.baidu.com/sug'
    s = input("输入你要查询的英文单词")
    dat = {
        "kw": s
    }
    response = requests.post(url, data=dat)  # post模拟用户发送请求，data是传输的kw参数
    # 数据参数传输必须用字典形式
    print(response.json())  # 服务器返回的内容直接转化成json
    response.close()

def xhl():  # 抓包工具F12的XHL选项一般为数据（在preview中查看），Img为图片
    # 找到对应的XHL后，在Header选项找URL，URL？前是链接，？后是参数
    # Request method是GET或POST，决定了爬虫是使用get还是post
    # Query String Parameters是请求时使用的参数
    param = {
        "type": "24",
        "interval_id": "100 % 3",
        "action": "",
        "start": "0",  # 每次start递增
        "limit": "20",
    }
    url = 'http://movie.douban.com/j/chart/top_list'
    response = requests.get(url, params=param)  # 请求服务器响应，post传参用datas，get传参用params
    print(response.request.headers)  # 'User-Agent': 'python-requests/2.27.1'表示没有模拟用户，被反爬虫了

    # 设置User-Agent进行模拟用户，值在header中找
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"}
    response = requests.get(url, params=param, headers=header)  # 模仿用户后，再次请求服务器响应
    print(response.json())  # 或response.text
    # 好像依然被反爬5555
    response.close()

if __name__ == '__main__':
    xhl()
