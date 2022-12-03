# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 00:24:11 2022

@author: 91749
"""

from PIL import Image
import os, sys

dir_path = "/home/ibanerj/Project/End_To_End_SelfDriving/dataset/valid"

def resize_im(path):
    if os.path.isfile(path):
        im = Image.open(path).resize((400,300), Image.ANTIALIAS)
        parent_dir = os.path.dirname(path)
        img_name = os.path.basename(path).split('.')[0]
        im.save(os.path.join(parent_dir, img_name + 'resize.png'), 'PNG', quality=90)

def resize_all(mydir):
    for subdir , _ , fileList in os.walk(mydir):
        for f in fileList:
            try:
                full_path = os.path.join(subdir,f)
                resize_im(full_path)
            except Exception as e:
                print('Unable to resize %s. Skipping.' % full_path)

if __name__ == '__main__':
    resize_all(dir_path)