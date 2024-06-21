'''
Date: 2024-02-14 16:08:08
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2024-06-21 15:14:19
Description: 
'''
import os
import random
import tkinter as tk
from typing import Dict, List, Tuple

from PIL import Image, ImageTk


class ImageMagnet(tk.Tk):
    def __init__(self, image_dir, window_size, min_image_size, scroll_direction):
        super().__init__()
        self.image_dir = image_dir
        self.window_size = window_size
        self.min_image_size = min_image_size
        self.scroll_direction = scroll_direction
        self.images = self.load_images()
        self.canvas = tk.Canvas(self, width=window_size[0], height=window_size[1])
        self.canvas.pack()
        self.show_images()

    def load_images(self):
        images = []
        for filename in os.listdir(self.image_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.image_dir, filename)
                image = Image.open(image_path)
                image = self.resize_image(image)
                images.append(ImageTk.PhotoImage(image))
        return images

    def resize_image(self, image):
        width, height = image.size
        if width < self.min_image_size[0] or height < self.min_image_size[1]:
            ratio = max(self.min_image_size[0] / width, self.min_image_size[1] / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)
        return image

    def show_images(self):
        if self.scroll_direction == "vertical":
            y = 0
            for image in self.images:
                self.canvas.create_image(0, y, anchor="nw", image=image)
                y += image.height()
        elif self.scroll_direction == "horizontal":
            x = 0
            for image in self.images:
                self.canvas.create_image(x, 0, anchor="nw", image=image)
                x += image.width()
        self.scroll_images()

    def scroll_images(self):
        if self.scroll_direction == "vertical":
            self.canvas.move("all", 0, -1)
            self.after(50, self.scroll_images)
        elif self.scroll_direction == "horizontal":
            self.canvas.move("all", -1, 0)
            self.after(50, self.scroll_images)

# if __name__ == "__main__":
#     image_dir = r"D:\^SYSTEM\SETTING\LiveWallper"
#     window_size = (800, 600)
#     min_image_size = (200, 200)
#     scroll_direction = "vertical"  # or "horizontal"
#     app = ImageMagnet(image_dir, window_size, min_image_size, scroll_direction)
#     app.attributes("-fullscreen", True)  # Uncomment this line for full screen
#     app.mainloop()
import numpy as np


class Box:
    def __init__(self, x: int = -1, y: int = -1,
                 w: int = 0, h: int = 0, boundarys: Dict[str, bool] = None) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        boundarys = boundarys or {"left": False, "right": False,
                                  "top": False, "bottom": False}
        self.boundarys: dict[str, bool] = boundarys
        
    def area(self) -> int:
        return self.w * self.h
    
    def copy(self):
        return Box(self.x, self.y, self.w, self.h, self.boundarys.copy())
    
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Box) and self.area() == __value.area()
    
    def __lt__(self, __value: object) -> bool:
        return isinstance(__value, Box) and self.area() < __value.area()

class BoxStatue:
    def __init__(self, big_box: Box, small_boxes: List[Box],
                 big_statue: np.ndarray[int],
                 small_statues: np.ndarray[bool]) -> None:
        """
        Parameters:
            - big_box: the big box that contains all small boxes.
            - small_boxes: Sorted with descending order of area. the small boxes that need to be put in the big box.
            - big_statue: the big box's statue, 0 means empty, others means occupied box's idx from samlle boxes list.
            - small_statues: the small boxes' statues, 0 means empty, 1 means occupied.
        """
        self.big_box = big_box
        self.small_boxes = small_boxes
        self.big_statue = big_statue
        self.small_statues = small_statues
        
    def _get_candidate_boxes(self) -> List[Box]:
        if self.small_statues.sum() == 0:
            return self.small_boxes
        idx = self.small_statues.argmax()
        return self.small_boxes[idx:]
    
    def _is_putable(self, box: Box, put_pos: Tuple[int, int]) -> bool:
        space = self.big_statue[put_pos[1]:put_pos[1]+box.h, put_pos[0]:put_pos[0]+box.w]
        if space.sum() == 0:
            if box.w < space.shape[1] and box.h < space.shape[0]:
                return box.copy()
            else:
                scale = min(box.w / space.shape[1], box.h / space.shape[0])
                new_w = int(box.w / scale)
                new_h = int(box.h / scale)
                return Box(put_pos[0], put_pos[1], new_w, new_h, self.big_box.boundarys.copy())
        else:
            new_w = space.sum(axis=-1, keepdims=False).argmin()
            new_h = space.sum(axis=0, keepdims=False).argmin()
            return Box(put_pos[0], put_pos[1], new_w, new_h, self.big_box.boundarys.copy())
        
    def put(self):
        candidate_boxes = self._get_candidate_boxes()
        if not candidate_boxes:
            return False
        box = candidate_boxes[0]