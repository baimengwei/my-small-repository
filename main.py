import csv
import requests
import time
import os

def storFile(data,fileName='0.csv',method='a'):
    with open(fileName,method,newline ='') as f:
        mywrite = csv.writer(f)
        mywrite.writerow(data)
        pass
    pass

for count_file in range(21):
    data_file=open(str(count_file)+'.csv','r',encoding='utf-8')
    data_list=data_file.readlines()
    data_file.close()
    count_ip=0
    for each in data_list:
        value=each.split(',')
        proxy=str(value[0])+':'+str(value[1].strip())
        proxies = {
          'http':proxy,
        }
        try:
            status=requests.get('http://www.baidu.com',proxies=proxies,timeout=1).status_code
            if status==200:
                count_ip=count_ip+1
                print(str(count_file)+'  '+str(count_ip)+'success ip:',proxy)
                storFile(proxy.split(':'),fileName=str(count_file+1)+'.csv')
            else:
                print(str(count_ip)+'error code:',status)
        except:
            print(str(count_ip)+'except, ip:',proxy)
for i in range(20):
    os.remove(str(i+1)+'.csv')
os.rename('21.csv','valid.csv')

