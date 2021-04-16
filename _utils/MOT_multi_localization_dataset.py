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


try:
    sys.path.insert(0,os.getcwd)
    from detrac.detrac_plot import pil_to_cv, plot_bboxes_2d
except:
    from detrac_plot import pil_to_cv, plot_bboxes_2d,plot_text



class Multi_Localization_Dataset(data.Dataset):
    """
    Creates an object for referencing the UA-Detrac 2D object tracking dataset
    and returning single object images for localization. Note that this dataset
    does not automatically separate training and validation data, so you'll 
    need to partition data manually by separate directories
    """
    
    def __init__(self, directory,mode = "train",cropsize = 224):
        """ initializes object
        image dir - (string) - a directory containing a subdirectory for each track sequence
        label dir - (string) - a directory containing a label file per sequence
        """
        self.classes = 1
        self.cs = int(cropsize)
        self.mode = mode
        
        self.class_dict = {
            'Pedestrian': 1,
            'Person on vehicle': 2,
            'Car': 3,
            'Bicycle': 4,
            'Motorbike': 5,
            'Non motorized vehicle': 6,
            'Static person': 7,
            'Distractor': 8,
            'Occluder': 9,
            'Occluder on the ground': 10,
            'Occluder full': 11,
            'Reflection': 12,
            'None' : 13,
            'Error':0,
            
            1:'Pedestrian',
            2:'Person on vehicle',
            3:'Car',
            4:'Bicycle',
            5:'Motorbike',
            6:'Non motorized vehicle',
            7:'Static person',
            8:'Distractor',
            9:'Occluder',
            10:'Occluder on the ground',
            11:'Occluder full',
            12:'Reflection',
            13:'None',
            0:'Error'
            }
            
        
        self.im_tf = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(brightness = 0.6,contrast = 0.6,saturation = 0.5)
                        ]),
                transforms.ToTensor(),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.07), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),

                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])

        # for denormalizing
        self.denorm = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                           std = [1/0.229, 1/0.224, 1/0.225])
        
        # track_list = [os.path.join(directory,sub_dir) for sub_dir in os.listdir(directory)]
        # track_list.sort()
        self.all_data = []

        # stores files for each image
        for sub_dir in os.listdir(directory):
            
            frame_dict = {}

            
            if mode == "train":
                # read lines from label file
                with open(os.path.join(directory,sub_dir,"gt","gt.txt")) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
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
                        bbox = torch.tensor([left,top,left+width,top+height]).int()
                        
                        if frame not in frame_dict.keys():
                            frame_dict[frame] = []
                        
                        frame_dict[frame].append([bbox,cls,conf,visibility,obj_idx])

            
            im_dir = os.path.join(directory,sub_dir,"img1")
            
            for item in os.listdir(im_dir):
                impath = os.path.join(im_dir,item)
                
                idx = int(item.split(".jpg")[0])
                
                if mode == "train":
                    self.all_data.append((frame_dict[idx],impath))
                else:
                    self.all_data.append(([],impath))
                
        


    def __len__(self):
        """ returns total number of frames in all tracks"""
        return len (self.all_data)
       # return self.total_num_frames
    
    def __getitem__(self,index):
        """ returns item indexed from all frames in all tracks from training
        or testing indices depending on mode
        """
    
        # load image and get label     
        label = []
        
        while len(label) == 0: # so all crops contain objects
            # avoid file open errors
            try:
                im.close()
                del im
            except:
                pass
            
            cur = self.all_data[index]
            im = Image.open(cur[1])
            label = cur[0]
            index = (index + 1) % len(self)
            
            if self.mode == "test":
                break
       
        
        
        if len(label) > 0:
            bboxes = []
            classes = []
            for item in label:
                bboxes.append(item[0])
                classes.append(item[1])
            bboxes = torch.stack(bboxes)
            classes = torch.stack(classes).unsqueeze(1)
            
            
            y = torch.cat((bboxes,classes),dim= 1).double()
            
        else:
            y = torch.zeros([1,5])
            
        # randomly flip
        FLIP = np.random.rand()
        if FLIP > 0.5:
            im= F.hflip(im)
            # reverse coords and also swit-ch xmin and xmax
            if torch.sum(y) != 0:
                new_y = torch.clone(y)
                new_y[:,0] = im.size[0] - y[:,2]
                new_y[:,2] = im.size[0] - y[:,0]
                y = new_y
          
            
        # use one object to define center
        if torch.sum(y) != 0:
            idx = np.random.randint(len(y))
            box = y[idx]
            centx = (box[0] + box[2])/2.0
            centy = (box[1] + box[3])/2.0
            noise = np.random.normal(0,20,size = 2)
            centx += noise[0]
            centy += noise[1]
            
            size = max(box[3]-box[1],box[2] - box[0])
            size_noise = max( -(size*2/4) , np.random.normal(size/2,size/4)) 
            size = size*1.25 + 0.3*size_noise # it's gonna be hard enough as it is
            
            if size < 50:
                size = 50
        else:
            size = max(50,np.random.normal(200,25))
            centx = im.size[0]/2
            centy = im.size[1]/2
        try:
            minx = int(centx - size/2)
            miny = int(centy - size/2)
            maxx = int(centx + size/2)
            maxy = int(centy + size/2)
        
        except TypeError:
            print(centx,centy,size)
        
        im_crop = F.crop(im,miny,minx,maxy-miny,maxx-minx)
        del im 
        
        if im_crop.size[0] == 0 or im_crop.size[1] == 0:
            print("Oh no! {} {} {}".format(centx,centy,size))
            raise Exception
            
        # shift labels if there is at least one object
   
        if torch.sum(y) != 0:
            y[:,0] = y[:,0] - minx
            y[:,1] = y[:,1] - miny
            y[:,2] = y[:,2] - minx
            y[:,3] = y[:,3] - miny

        crop_size = im_crop.size
        im_crop = F.resize(im_crop, (self.cs,self.cs))

        y[:,0] = y[:,0] * self.cs/crop_size[0]
        y[:,2] = y[:,2] * self.cs/crop_size[0]
        y[:,1] = y[:,1] * self.cs/crop_size[1]
        y[:,3] = y[:,3] * self.cs/crop_size[1]
        
        # remove all labels that aren't in crop
        if torch.sum(y) != 0:
            keepers = []
            for i,item in enumerate(y):
                if item[0] < self.cs-15 and item[2] > 0+15 and item[1] < self.cs-15 and item[3] > 0+15:
                    keepers.append(i)
            y = y[keepers]
        if len(y) == 0:
            y = torch.zeros([1,5])
            y[0,4] = 0
        
        im_t = self.im_tf(im_crop)    
        
        return im_t, y
        
    
    def num_classes(self):
        return self.classes
    
    def show(self,index):
        """ plots all frames in track_idx as video
            SHOW_LABELS - if True, labels are plotted on sequence
            track_idx - int    
        """
        mean = np.array([0.485, 0.456, 0.406])
        stddev = np.array([0.229, 0.224, 0.225])
        
        im,label = self[index]
        
        im = self.denorm(im)
        cv_im = np.array(im) 
        cv_im = np.clip(cv_im, 0, 1)
        
        # Convert RGB to BGR 
        cv_im = cv_im[::-1, :, :]         
        
        cv_im = np.moveaxis(cv_im,[0,1,2],[2,0,1])

        cv_im = cv_im.copy()

        class_colors = [
            (255,150,0),
            (255,100,0),
            (255,50,0),
            (0,255,150),
            (0,255,100),
            (0,255,50),
            (0,100,255),
            (0,50,255),
            (255,150,0),
            (255,100,0),
            (255,50,0),
            (0,255,150),
            (0,255,100),
            (0,255,50),
            (0,100,255),
            (0,50,255),
            (200,200,200) #ignored regions
            ]
        
        
        for bbox in label:
            bbox = bbox.int().data.numpy()
            cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[bbox[4]], 1)
            plot_text(cv_im,(bbox[0],bbox[1]),bbox[4],0,class_colors,self.class_dict)
        
        
        # for region in metadata["ignored_regions"]:
        #     bbox = region.astype(int)
        #     cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[-1], 1)
       
        #cv_im = cv2.resize(cv_im,(400,400))
        cv2.imshow("Frame",cv_im)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

def collate(inputs):
    """
    Recieves list of tuples and returns a tensor for each item in tuple, except metadata
    which is returned as a single list
    """
    im = [] # in this dataset, always [3 x W x H]
    label = [] # variable length
    max_labels = 0
    largest_h = 0
    largest_w = 0
    
    for batch_item in inputs:
        im.append(batch_item[0])
        label.append(batch_item[1])
        
        # since images aren't all the same size, keep track of largest image size
        if batch_item[0].shape[2] > largest_w:
            largest_w = batch_item[0].shape[2]
        if batch_item[0].shape[1] > largest_h:
            largest_h = batch_item[0].shape[1]
        
        # keep track of image with largest number of annotations
        if len(batch_item[1]) > max_labels:
            max_labels = len(batch_item[1])
        
    # pad all images to max size
    pad_ims = []
    for image in im:
        new = torch.zeros([3,largest_h,largest_w])
        new[:,:image.shape[1],:image.shape[2]] = image
        pad_ims.append(new)
    # collate images        
    ims = torch.stack(pad_ims)
    
    # collate labels
    labels = torch.zeros([len(label),max_labels,5]) - 1
    for idx in range(len(label)):
        num_objs = len(label[idx])
        
        labels[idx,:num_objs,:] = label[idx]
        
    return ims,labels
     

if __name__ == "__main__":
    #### Test script here

    directory = "/home/worklab/Data/cv/MOT20/train"
    test = Multi_Localization_Dataset(directory,mode = "train",cropsize = 224/2)
    for i in range(10):
        idx = np.random.randint(0,len(test))
        test.show(idx)
        test.show(idx)
    
    cv2.destroyAllWindows()