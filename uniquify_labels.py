#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:06:56 2021

@author: worklab
"""


import numpy as np
import os
import cv2
import csv
import copy
import torch
import argparse
import string
import _pickle as pickle
import cv2 as cv
from PIL import Image
from torchvision.transforms import functional as F


def expand_labels(directory):
    for file in os.listdir(directory):
        file = os.path.join(directory,file)
        
        # p = int(file.split("record_")[1][1])
        # c = int(file.split("record_")[1][3])
        # n = int(file.split("record")[1].split("_")[2])
        p = int(file.split("/")[-1][1])
        c = int(file.split("/")[-1][3])
        n = int(file.split("/")[-1][5])
        
        pcn = int(str(p) + str(c) + str(n))
        
        #print(pcn)
        output_rows = []
        with open(file,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                
                
                if HEADERS:
                    if len(row) > 0 and row[0][0:5] == "Frame":
                        HEADERS = False # all header lines have been read at this point
                else:
                    # see if there is a 3D bbox associated with that object and frame
                    obj_idx = str(row[2])
                    
                    row[2] = int(obj_idx + str(pcn))
                        
                output_rows.append(row)
    
        # write final output file
        outfile = file
        
        with open(outfile, mode='w') as f:
            out = csv.writer(f, delimiter=',')
            out.writerows(output_rows)
        
        print("Wrote unique object indexes for file {}".format(file.split("/")[-1]))
    
    
    
    
    
if __name__ == "__main__":
    directory = "/home/worklab/Data/dataset_alpha/track_2d_unique"
    expand_labels(directory)