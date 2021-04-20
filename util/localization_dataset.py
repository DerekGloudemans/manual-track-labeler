"""
This file provides a dataset class for working with the UA-detrac tracking dataset.
Provides:
    - plotting of 2D bounding boxes
    - training/testing loader mode (random images from across all tracks) using __getitem__()
    - track mode - returns a single image, in order, using __next__()
"""

import os,sys
import numpy as np
import random 
import pandas as pd
import json

import torch
import torchvision.transforms.functional as F
    
import cv2
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms




def pil_to_cv(pil_im):
    """ convert PIL image to cv2 image"""
    open_cv_image = np.array(pil_im) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1] 


def plot_text(im,offset,cls,idnum,class_colors,class_dict):
    """ Plots filled text box on original image, 
        utility function for plot_bboxes_2
        im - cv2 image
        offset - to upper left corner of bbox above which text is to be plotted
        cls - string
        class_colors - list of 3 tuples of ints in range (0,255)
        class_dict - dictionary that converts class strings to ints and vice versa
    """

    text = "{}: {}".format(idnum,class_dict[cls])
    
    font_scale = 2.0
    font = cv2.FONT_HERSHEY_PLAIN
    
    # set the rectangle background to white
    rectangle_bgr = class_colors[cls]
    
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    
    # set the text start position
    text_offset_x = int(offset[0])
    text_offset_y = int(offset[1])
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(im, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(im, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0., 0., 0.), thickness=2)
    

class Localization_Dataset(data.Dataset):
    """
    Blah Blah Blah ....
    """
    
    def __init__(self, image_dir, label_file, mode = "training"):
        """ initializes object. By default, the first track (cur_track = 0) is loaded 
        such that next(object) will pull next frame from the first track
        image dir - (string) - a directory containing a subdirectory for each track sequence
        label file - (string) - 
        mode - (string) - training or testing
        """
        
        
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
        
        self.class_dict = {
            "sedan": 0,
            "SUV":1,
            "minivan":2,
            "van":3,
            "pickup truck": 4,
            "pickup":4,
            "semi":5,
            "semi truck": 5,
            "truck (other)": 6,
            "trailer":7,
            "motorcycle":8,
            0:"sedan",
            1:"SUV",
            2:"minivan",
            3:"van",
            4:"pickup truck",
            5:"semi truck",
            6:"truck (other)",
            7:"trailer",
            8:"motorcycle"
            
            
            }
        
        self.labels = []
        self.data = []
        
        df = pd.read_csv(label_file)
        im_names = df['filename'].unique()
        im_names = sorted(im_names)
        
        # get all data for a given image
        for item in im_names:
            rows = df[df.filename == item]
            rows = rows.to_numpy()
            
            gathered = []
            try:
                for row in rows:
                    bbox = json.loads(row[5])
                    if bool(bbox): # not empty
                        bbox = [bbox["x"],bbox["y"],bbox["width"],bbox["height"]]
                        cls = json.loads(row[6])["class"]
                        bbox.append(self.class_dict[cls])
                        bbox = np.array(bbox)
                        gathered.append(bbox)
            except:
                pass
            
            gathered = np.array(gathered)
            self.labels.append(gathered)
            self.data.append(os.path.join(image_dir,item))
            
        
        indices = [i for i in range(len(self.labels))]
        random.seed(5)
        random.shuffle(indices)
        
        if mode == "training":
            indices = indices[:int(0.9*len(indices))]
        else:
            indices = indices[int(0.9*len(indices)):]
            
        labels = [self.labels[i] for i in indices]
        data =   [self.data[i] for i in indices]
        
        self.labels = labels
        self.data = data
        
    
    def __getitem__(self,index):
        """ returns item indexed from all frames in all tracks from training
        or testing indices depending on mode
        """
        
        # load image and get label        
        label = self.labels[index]
        im = Image.open(self.data[index])
        #im = F.resize(im, (1080,1920))
        
        y = torch.from_numpy(label)
        
        if y.numel() == 0:
            y = torch.zeros([1,5])
            
            
        
        yn = y.clone()
        yn[:,2] = y[:,0] + y[:,2]
        yn[:,3] = y[:,1] + y[:,3]
        y = yn
        
        #y[:,:4] = y[:,:4]/2.0 # due to resize
        
        # randomly flip
        FLIP = np.random.rand()
        if FLIP > 0.5:
            im= F.hflip(im)
            # reverse coords and also switch xmin and xmax
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
            
            size = max(box[3]-box[1],box[2] - box[0]) * 2
            size_noise = min(max( -(size*6/4) , np.random.normal(0,size/2)),size)
            size_noise = 0
            size += size_noise
            
            if size < 50:
                size = 50
        else:
            size = max(200,np.random.normal(500,100))
            centx = np.random.randint(200,1900)
            centy = np.random.randint(200,3500)
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
        im_crop = F.resize(im_crop, (224,224))

        y[:,0] = y[:,0] * 224/crop_size[0]
        y[:,2] = y[:,2] * 224/crop_size[0]
        y[:,1] = y[:,1] * 224/crop_size[1]
        y[:,3] = y[:,3] * 224/crop_size[1]
        
        # remove all labels that aren't in crop
        if torch.sum(y) != 0:
            keepers = []
            for i,item in enumerate(y):
                if item[0] < 224-15 and item[2] > 0+15 and item[1] < 224-15 and item[3] > 0+15:
                    keepers.append(i)
            y = y[keepers]
        if len(y) == 0:
            y = torch.zeros([1,5])
            y[0,4] = -1
        
        im_t = self.im_tf(im_crop)
        
        
        return im_t, y
    
    
    def __len__(self):
        return len(self.labels)
    
    def label_to_name(self,num):
        return self.class_dict[num]
        
    def load_annotations(self,idx):
        """
        Loads labels in format for mAP evaluation 
        list of arrays, one [n_detections x 4] array per class
        """
        annotation = [[] for i in range(self.classes)]
        
        label = self.all_data[idx][1]
        
        for obj in label:
            cls = int(obj['class_num'])
            annotation[cls].append(obj['bbox'].astype(float))
        
        for idx in range(len(annotation)):
            if len(annotation[idx]) > 0:
                annotation[idx] = np.stack(annotation[idx])
            else:
                annotation[idx] = np.empty(0)
        return annotation
    
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
       
        #cv_im = cv2.resize(cv_im,(1920,1080))
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
    
    for batch_item in inputs:
        im.append(batch_item[0])
        label.append(batch_item[1])
        
        # keep track of image with largest number of annotations
        if len(batch_item[1]) > max_labels:
            max_labels = len(batch_item[1])
        
    # collate images        
    ims = torch.stack(im)
    
    # collate labels
    labels = torch.zeros([len(label),max_labels,5]) - 1
    for idx in range(len(label)):
        num_objs = len(label[idx])
        
        labels[idx,:num_objs,:] = label[idx]
        
    return ims,labels
        

if __name__ == "__main__":
    #### Test script here

    label_dir = "/home/worklab/Data/cv/i24_2D_October_2020/labels.csv"
    image_dir = "/home/worklab/Data/cv/i24_2D_October_2020/ims"
    test = Localization_Dataset(image_dir,label_dir)
    for i in range(100):
        test.show(i)
        temp = test[0]

