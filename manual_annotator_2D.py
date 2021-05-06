# using openCV, draw ignored regions for each frame. 

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


class Annotator_2D():
    
    def __init__(self,sequence, ds = 1,load_corrected = False):
        """
        sequence - path to video file
        """
        
        self.sequence_path = sequence
        self.label_path = sequence.split(".mp4")[0] + "_track_outputs.csv"
        if load_corrected:
            self.label_path = sequence.split(".mp4")[0] + "_track_outputs_corrected.csv"
        self.ds = ds
        
        # open VideoCapture
        self.cap = cv2.VideoCapture(sequence)
        ret,frame = self.cap.read()
        if self.ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
        self.frame_buffer = [frame]
        self.cur_frame = frame.copy()
        
        self.buffer_frame_num = 0
        self.frame_num = 0        
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.start_point = None # used to store click temporarily
        self.clicked = False 
        self.new = None # used to store a new box to be plotted temporarily
        self.cont = True
        self.active_command = None
        self.colors = (np.random.rand(100,3))*255
        self.last_active_obj_idx = -1
        
        # load labels
        frame_labels = {}
        with open(self.label_path,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                
                if not HEADERS:
                    frame_idx = int(row[0])
                    
                    if frame_idx not in frame_labels.keys():
                        frame_labels[frame_idx] = [row]
                    else:
                        frame_labels[frame_idx].append(row)
                        
                if HEADERS and len(row) > 0:
                    if row[0][0:5] == "Frame":
                        HEADERS = False # all header lines have been read at this point
                        
        
        self.labels = frame_labels
        self.label_buffer = [copy.deepcopy(frame_labels)]
    
        self.plot()
        
    
    def on_mouse(self,event, x, y, flags, params):
    
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         self.start_point = (x,y)
         self.clicked = True
         self.changed = True
         
       elif event == cv.EVENT_LBUTTONUP:
            box = np.array([self.start_point[0],self.start_point[1],x,y])
            self.new = box
            self.clicked = False
              
                 
    def plot(self):
        """
        Replots current frame
        """
        
        cur_frame_objs = self.labels[self.frame_num].copy()
    
        index_in_buffer = -(self.buffer_frame_num - self.frame_num) - 1
        self.cur_frame = self.frame_buffer[index_in_buffer].copy()
        
        for box in cur_frame_objs:
            cls = box[3]
            oidx = int(box[2])
            bbox = np.array(box[4:8]).astype(float).astype(int)
            color = self.colors[oidx%100]
            
            label = "{} {}".format(cls,oidx)
            self.cur_frame = cv2.rectangle(self.cur_frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,2)
            self.cur_frame = cv2.putText(self.cur_frame,"{}".format(label),(int(bbox[0]),int(bbox[1] - 10)),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),3)
            self.cur_frame = cv2.putText(self.cur_frame,"{}".format(label),(int(bbox[0]),int(bbox[1] - 10)),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1)
    
    def delete(self,obj_idx):
        """
        Delete object obj_idx in this and all subsequent frames
        """
        frame_idx = self.frame_num 
        
        while frame_idx < self.length:
            try:
                for idx,row in enumerate(self.labels[frame_idx]):
                    if int(row[2]) == obj_idx:
                        del self.labels[frame_idx][idx]
                        break
            except KeyError:
                pass
            frame_idx += 1
        
        print("Deleted obj {} in frame {} and all subsequent frames".format(obj_idx,self.frame_num))
    
    def add(self,obj_idx,bbox):
        """
        Add a new box with the specified obj_idx - we find another box with the same obj_idx, and copy the class
        """
        cls = None
        vel_x = 0
        vel_y = 0
        for frame in self.labels:
            labels = self.labels[frame]
            for box in labels:
                if int(box[2]) == obj_idx:
                    cls = box[3]
                    vel_x = float(box[8])
                    vel_y = float(box[9])
                    break
            if cls is not None:
                break
        
        if cls is None:
            cls = input("This object does not exist elsewhere. Input class:")
        
        timestamp = ""
        for box in self.labels[self.frame_num]:
            if len(box[1]) > 0:
                timestamp = box[1]
                break
        
        new_row = [self.frame_num,timestamp,obj_idx,cls,bbox[0],bbox[1],bbox[2],bbox[3],vel_x,vel_y]
        print(new_row)
        self.labels[self.frame_num].append(new_row)
        
        print("Added box for object {} to frame {}".format(obj_idx,self.frame_num))
        
        
    def reassign(self,old,new):
        """
        Assign the new obj_idx to the old obj_idx object for this and subsequent frames
        """
        frame_idx = self.frame_num 
        
        while frame_idx < self.length:
            try:
                for idx,row in enumerate(self.labels[frame_idx]):
                    if int(row[2]) == old:
                        row[2] = new
            except KeyError:
                pass
            frame_idx += 1
        
        print("Changed obj {} to obj {} in frame {} and all subsequent frames".format(old,new,self.frame_num))
        
    
    def redraw(self,obj_idx,box):
        """
        
        """
        
        for row in self.labels[self.frame_num]:
            if int(row[2]) == obj_idx:
                row[4:8] = box
                
        print("Redrew box {} for frame {}".format(obj_idx,self.frame_num))
        
    
    def undo(self):
        """
        undo the last operation for this frame 
        """
        if len(self.label_buffer) > 1:
            
            self.labels = self.label_buffer[-2].copy()
            self.label_buffer = self.label_buffer[:-1]
            
            self.plot()
            print("Undid last operation")
        
        else:
            print("Can't undo the last operation")
       
            

    def find_box(self,point):
        point = point *2
        print(point)

        for row in self.labels[self.frame_num]:
            bbox = np.array(row[4:8]).astype(float).astype(int)
             # since we downsample image before plotting
            print(bbox)
            if point[0] > bbox[0] and point[0] < bbox[2] and point[1] > bbox[1] and point[1] < bbox[3]:
                obj_idx = int(row[2])
                return obj_idx
        
        # determine whether point falls within any object boxes
        return -1
        
    def next(self):
        """
        Advance to next frame
        """
        self.clicked = False
        
        if self.frame_num < self.length - 1:
            
            if self.frame_num == self.buffer_frame_num:
                ret,frame = self.cap.read()
                if self.ds != 1:
                    frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
                self.frame_buffer.append(frame.copy())
                self.cur_frame = frame.copy()
                
                if len(self.frame_buffer) > 30:
                    self.frame_buffer = self.frame_buffer[1:]
                
                self.frame_num += 1
                self.buffer_frame_num += 1
            
                self.plot()
            
            else:
                # get frame frome frame_buffer
                self.frame_num += 1
                self.plot()
            
            
        else:
            print("On last frame, cannot advance to next frame")
        
        
        
    def prev(self):
        """
        Return to previous frame, which is stored in a temporary buffer
        """
        self.clicked = False
        if self.frame_num > 0:
            self.frame_num -= 1
            
            self.plot()
            
        else:
            print("On first frame, cannot return to previous frame")
    
    def save(self):
        """
        Saves the current annotations to CSV 
        """
        
        # load csv file annotations into list
        output_rows = []
        with open(self.label_path,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                
                if HEADERS:
                    if len(row) > 0 and row[0][0:5] == "Frame":
                        HEADERS = False # all header lines have been read at this point
                        row = row 
                        break
                output_rows.append(row)

        frames = self.labels.keys()
        frames.sort()
        
        for frame in frames:
            for row in self.labels[frame]:
                output_rows.append(row)
    
        # write final output file
        outfile = label_file.split(".csv")[0] + "_corrected.csv"
        with open(outfile, mode='w') as f:
            out = csv.writer(f, delimiter=',')
            out.writerows(output_rows)
            
        print("Wrote output rows")
       
    def quit(self):
        cv2.destroyAllWindows()
        self.cont = False
        self.save()

    def keyboard_input(self):
        keys = ""
        letters = string.ascii_lowercase + string.digits
        while True:
            key = cv2.waitKey(1)
            for letter in letters:
                if key == ord(letter):
                    keys = keys + letter
            if key == ord("\n") or key == ord("\r"):
                break
        return keys
              
    def run(self):
        """
        Main processing loop
        """
        
        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
           
        while(self.cont): # one frame
        
           # handle click actions
           if self.new is not None:
               
                if self.active_command == "DELETE":
                    obj_idx = self.find_box(self.new)
                    self.delete(obj_idx)

                elif self.active_command == "ADD":
                    # get obj_idx
                    try:
                        obj_idx = int(self.keyboard_input())
                        self.last_active_obj_idx = obj_idx
                    except:
                        obj_idx = self.last_active_obj_idx
                    
                    
                    self.new *= 2
                    self.add(obj_idx,self.new)
                
                elif self.active_command == "REASSIGN":
                    old_obj_idx = self.find_box(self.new)
                    
                    try:
                        obj_idx = int(self.keyboard_input())
                        self.last_active_obj_idx = obj_idx
                    except:
                        obj_idx = self.last_active_obj_idx
                   
                    self.reassign(old_obj_idx,obj_idx)
                    
                elif self.active_command == "REDRAW":
                    center = np.array([(self.new[0] + self.new[2])/2.0,(self.new[1]+self.new[3])/2.0])
                    print(center)
                    obj_idx = self.find_box(center)
                    self.new *= 2
                    self.redraw(obj_idx,self.new)
                    
                self.label_buffer.append(copy.deepcopy(self.labels))
                if len(self.label_buffer) > 50:
                    self.label_buffer = self.label_buffer[1:]
                    
                self.plot()
                self.new = None     
           
               
           self.cur_frame = cv2.resize(self.cur_frame,(1920,1080))
           cv2.imshow("window", self.cur_frame)
           title = "Frame {}/{} --- Active Command: {}".format(self.frame_num,self.length,self.active_command)
           cv2.setWindowTitle("window",str(title))
           
           key = cv2.waitKey(1)
           if key == ord('9'):
                self.next()
           elif key == ord('8'):
                self.prev()
           
           elif key == ord('d'):
                self.active_command = "DELETE"
           elif key == ord("r"):
                self.active_command = "REASSIGN"
           elif key == ord("a"):
                self.active_command = "ADD"
           elif key == ord("w"):
                self.active_command = "REDRAW"
           
           elif key == ord("q"):
                self.quit()
           elif key == ord("u"):
               self.undo()
           
        
        
        
if __name__ == "__main__":
    
    test_path = "/home/worklab/Data/cv/3D examples for Dan/record_p1c2_00000.mp4"
    ann = Annotator_2D(test_path,load_corrected = False)
    ann.run()