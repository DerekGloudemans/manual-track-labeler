"""
Derek Gloudemans - August 4, 2020
Adapted from https://github.com/yhenon/pytorch-retinanet - train.py

This file provides a dataset class for working with the MOT challenge tracking dataset.
Provides:
    - plotting of 2D bounding boxes
    - training/testing loader mode (random images from across all tracks) using __getitem__()
    - track mode - returns a single image, in order, using __next__()
"""

import os,sys
import numpy as np
import random 
import csv
random.seed = 0

import cv2
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
from scipy.signal import savgol_filter


try:
    sys.path.insert(0,os.getcwd)
    from detrac.detrac_plot import pil_to_cv, plot_bboxes_2d
except:
    from detrac_plot import pil_to_cv, plot_bboxes_2d,plot_text



class Tracking_Dataset(data.Dataset):
    """
    Creates an object for referencing the UA-Detrac 2D object tracking dataset
    and returning single object images for localization. Note that this dataset
    does not automatically separate training and validation data, so you'll 
    need to partition data manually by separate directories
    """
    
    def __init__(self, directory,n = 15):
        """ initializes object
        image dir - (string) - a directory containing a subdirectory for each track sequence
        label dir - (string) - a directory containing a label file per sequence
        """
        
        
            
        self.n = n
        self.im_list = []
        self.label_list = []

        # stores files for each image
        for sub_dir in os.listdir(directory):
            
            obj_dict = {}

            # read lines from label file
            with open(os.path.join(directory,sub_dir,"gt","gt.txt")) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader: # each row corresponds to one frame and object
                    frame,obj_idx,left,top,width,height,conf,cls,visibility = row
                    frame = int(frame)
                    obj_idx = int(obj_idx)
                    left = int(left)
                    top = int(top)
                    width = int(width)
                    height = int(height)
                    conf = float(conf)
                    cls = torch.tensor(int(cls)).int()
                    visibility = float(visibility)

                        
                    datum = torch.tensor([left,top,left+width,top+height])
                    im_path = os.path.join(directory,sub_dir,"img1",str(frame).zfill(6)+".jpg")
                    
                    if obj_idx in obj_dict:
                        obj_dict[obj_idx][0].append(datum)
                        obj_dict[obj_idx][1].append(im_path)
                    else:
                        obj_dict[obj_idx] = [[datum],[im_path]]
            
            for key in obj_dict.keys():
                tracklet = obj_dict[key]
                

                if len(tracklet[0]) > 7:
                    self.im_list.append(tracklet[1])

                    label = torch.stack(tracklet[0])
                    self.label_list.append(label)

              
            
    
        # parse_labels returns a list (one frame per index) of lists, where 
        # each item in the sublist is one object
        # so we need to go through and keep a running record of all objects, indexed by id
            
        with_speed = []
        for bboxes in self.label_list:

            speeds = np.zeros(bboxes.shape)  
            speeds[:len(speeds)-1,:] = bboxes[1:,:] - bboxes[:len(bboxes)-1,:]
            speeds[-1,:] = speeds[-2,:]

            try:
                speeds = savgol_filter(speeds,5,2,axis = 0)
            except:
                print(speeds.shape)
                print(bboxes.shape)
            #plt.plot(speeds[:,0])
            #plt.legend(["Unsmoothed","Smoothed"])
            #plt.show()
            combined = np.concatenate((bboxes,speeds),axis = 1)
            with_speed.append(combined)
        self.label_list = with_speed
                
        


    def __len__(self):
        """ returns total number of frames in all tracks"""
        return len (self.label_list)
       # return self.total_num_frames
    
    def __getitem__(self, index):
        
        data = self.label_list[index]

        # if track is too short, just use the next index instead
        while len(data) <= self.n:
            index = (index + 1) % len(self.label_list)
            data = self.label_list[index]
        
        start = np.random.randint(0,len(data)-self.n)
        data = data[start:start+self.n,:]
        
        ims = self.im_list[index]
        ims = ims[start:start+self.n]
        
        return data, ims
            
    
    
     

if __name__ == "__main__":
    #### Test script here

    directory = "/home/worklab/Data/cv/MOT20/train"
    test = Tracking_Dataset(directory,n = 12)
    for i in range(10):
        idx = np.random.randint(0,len(test))
        temp = test[idx]
        print (temp[0].shape)
    cv2.destroyAllWindows()