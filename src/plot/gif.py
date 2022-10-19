'''
Author: BHM-Bob 2262029386@qq.com
Date: 2022-10-19 23:02:16
LastEditors: BHM-Bob
LastEditTime: 2022-10-19 23:02:24
Description: 
'''
import glob

import imageio


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

def test():
    image_list = glob.glob(r'E:\My_Progs\z_Progs_Data_HC\Python_Utils\in\*.jpg')
    gif_name = r'E:\My_Progs\z_Progs_Data_HC\Python_Utils\out\out.gif'
    duration = 0.35
    create_gif(image_list, gif_name, duration)