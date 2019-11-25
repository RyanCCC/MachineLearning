import requests
import urllib.request
import urllib.error

'''
URLERROR:本地没有网络；无法连接；触发HTTPError
'''

if __name__ == '__main__':
    try:
        url ="https://blog.csdn.net/zzwu?viewmode=contents"
        agent_value = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36"
        # 伪装成一个浏览器
        header = {"User-Agent":agent_value}
        response = requests.get(url, headers=header)
        with open('test1.html', 'wb') as f:
            f.write(response.text.encode())
        header = ("User-Agent", agent_value)
        opener = urllib.request.build_opener()
        opener.addheaders=[header]
        response = opener.open(url).read()
        with open('./test.html', 'wb') as f:
            f.write(response)
        response = urllib.request.urlopen(url)
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code) 
        elif hasattr(e, "reason"):
            print(e.reason)
        else:
            print(e)
    print(response)
    print("hello world")