'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-10-19 23:03:16
LastEditors: BHM-Bob
LastEditTime: 2022-10-19 23:03:25
Description: 
'''
import segno


def MakeQRCode(url = r'http://www.lzu.edu.cn/',outputPath = 'imgs/qr.gif', backgurandPath = None,scale = 8):
    """
    """
    qrcode = segno.make(url, error='h')
    if backgurandPath == None:
        qrcode.save(outputPath)
    else:
        qrcode.to_artistic(background=backgurandPath, target=outputPath, scale=8)