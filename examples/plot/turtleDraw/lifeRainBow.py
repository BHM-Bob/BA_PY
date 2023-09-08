'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-10-19 22:48:27
LastEditors: BHM-Bob
LastEditTime: 2022-10-19 22:52:45
Description: 
'''
import math
import time
import turtle

Rq = 300#半圆的半径，正方形边长的一半
maxLifeLong = 100
scale = 30
lifeSeq = {
    '子女':[ [1, 0], [0.8, 4], [0.6, 8], [0.5, 16], [0.4, 19], [0.5, 30], [0.9, 40], [0.9, 70] ],
    '学生':[ [0.2, 4], [0.4, 8], [0.5, 16], [0.4, 19], [0.8, 23], [0.5, 30], [0.4, 60], [0.4, 85] ],
    '公民':[ [0.4, 18], [0.5, 20], [0.7, 30], [0.5, 55], [0.4, 80], [0.4, 85] ],
    '休闲者':[ [0.2, 8], [0.1, 17], [0.5, 19], [0.4, 20], [0.2, 22], [0.5, 30], [0.7, 70], [0.5, 85] ],
    '工作者':[ [0.2, 25], [0.8, 30], [0.6, 50], [0.2, 65], [0.2, 70] ],
    '持家者':[ [0.2, 25], [0.5, 30], [0.6, 50], [0.7, 85], [0.7, 85] ]
}
colorSeq = {
    '子女':[ 0.9, 0.3, 0.3 ],
    '学生':[ 0.0, 0.1, 0.9 ],
    '公民':[ 0.9, 0.9, 0.0 ],
    '休闲者':[ 0.0, 0.9, 0.5 ],
    '工作者':[ 0.9, 0.6, 0.0 ],
    '持家者':[ 0.6, 0.2, 0.9 ],
}

def MoveTo(x,y,angle, penDown = False):
    if not penDown:
        turtle.penup()
    turtle.goto(x,y)
    turtle.seth(angle)
    if not penDown:
        turtle.pendown()

def DrawCurve(startAge, endAge, R, penDown = False):
    """
    移动画笔至弧形起点，垂直于半径，结束后画笔朝向圆心
    """
    angle = 180 * (0.5 - startAge/maxLifeLong)#画弧起始画笔朝向
    drawAngle = 180 * (endAge - startAge) / maxLifeLong#画弧度数
    MoveTo( R*math.cos(math.pi*(90+angle)/180), R*math.sin(math.pi*(90+angle)/180), angle, penDown)
    turtle.circle(-R, drawAngle)
    turtle.right(90)

def DrawBar(r, seq):
    """
    seq: [相对高度，开始转变的年龄 ] => [ [0.9 , 0], [0.7 , 10], [0.9 , 40], [0.6 , 60] ]
    """
    #绘制底层弧形
    DrawCurve(seq[-1][1], seq[0][1], r)
    #绘制始端
    DrawCurve(seq[0][1], seq[0][1], r, True)
    turtle.backward(scale*seq[0][0])
    #绘制上层弧形
    for idx in range(len(seq) - 1):
        DrawCurve(seq[idx][1], seq[idx + 1][1], scale*seq[idx][0]+r, True)
        turtle.forward(scale*(seq[idx][0] - seq[idx + 1][0]))
    #绘制末端
    DrawCurve(seq[-1][1], seq[-1][1], scale*seq[-1][0]+r, True)
    turtle.forward(scale*seq[-1][0])

turtle.screensize(canvwidth=2*Rq, canvheight=2*Rq)
turtle.speed(10)
#绘制最外围半圆
DrawCurve(0, maxLifeLong, Rq)
turtle.forward(2*Rq)
#写年龄
for age in range(0,maxLifeLong,5):
    DrawCurve(age, age, Rq+scale/3)
    turtle.write(str(age),font=("Arial" , 10 , "normal"))
#绘制折形弧形
sR = 80#弧形半径
for key in lifeSeq.keys():
    #写标签
    DrawCurve(lifeSeq[key][0][1]-5, lifeSeq[key][0][1]-5, sR+scale)
    turtle.write(key,font=("Arial" , 10 , "normal"))
    #画弧
    turtle.fillcolor(colorSeq[key][0],colorSeq[key][1],colorSeq[key][2])
    turtle.begin_fill()
    DrawBar(sR, lifeSeq[key])
    turtle.end_fill()
    sR += scale*1.2

MoveTo(-100,-100,0)
turtle.write("生科2班 BHM-Bob G",font=("Arial" , 10 , "normal"))
time.sleep(999)