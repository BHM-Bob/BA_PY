'''
Author: BHM-Bob G 2262029386@qq.com
Date: 2022-04-27 16:26:32
LastEditors: BHM-Bob G
LastEditTime: 2022-07-12 16:20:24
Description: 
'''
from PIL import Image
from io import BytesIO
import requests

from ba.web import *

def getImg(taskQue, doneDataQue, sig, statuesQue):
    while True:
        idx, imgUrl = taskQue.get()
        if idx == -1:
            break
        BytesIOObj = BytesIO()
        response = requests.get(imgUrl).content
        BytesIOObj.write(response)
        doneDataQue.put((idx, Image.open(BytesIOObj).convert( "RGB" )))
    return sig.put(0)

imgHref = 'https://s3.ananas.chaoxing.com/doc/31/a1/b8/e181b5a08384a9df472773b8ab672508/thumb/110.png'
imgList = []
iscontinue = True
tp = ThreadsPool(8, getImg, statuesQue)
while iscontinue:
    lastPageLink = str(GetInput("input last png 's link:"))
    imgHref = '/'.join(lastPageLink.split('/')[:-1]) + '/'
    sumPages = int(lastPageLink.replace(imgHref,'').replace('.png','')) + 1
    name = lastPageLink.split('/')[-3]
    print('page url: {''}\nsum page(s): {:d}'.format(imgHref, sumPages-1))
    for page in range(1, sumPages):
        tp.PutTask((page, imgHref + str(page) + '.png'))
    dataList = tp.LoopToQuit((-1, None))
    dataList.sort(key = lambda x : x[0])
    imgList = [ i[1] for i in dataList ]
    imgList[0].save("E:/"+name+".pdf", "pdf", save_all=True, append_images=imgList[1:])
    iscontinue = GetInput("input 1 means continue, others means quit:") == '1'

