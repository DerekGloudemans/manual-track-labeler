# using openCV, draw ignored regions for each frame. 

import numpy as np
import os
import cv2
import csv
import copy
import argparse
import string

import cv2 as cv



class Annotator_2D():
    
    def __init__(self,sequence1,sequence2,label_dir, ds = 1):
        """
        sequence - path to video file
        """
        
        self.sequence1_path = sequence1
        self.sequence1_short_name = sequence1.split("/")[-1].split(".mp4")[0]
        self.label_path1 = os.path.join(label_dir,self.sequence1_short_name + "_track_outputs_corrected.csv")
        
        self.sequence2_path = sequence2
        self.sequence2_short_name = sequence2.split("/")[-1].split(".mp4")[0]
        self.label_path2 = os.path.join(label_dir,self.sequence2_short_name + "_track_outputs_corrected.csv")
        
        self.ds = ds
        self.plot_ds = 3
        
        
        # open VideoCapture
        self.cap1 = cv2.VideoCapture(sequence1)
        ret,frame = self.cap1.read()
        if self.ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
        
        self.frame_buffer1 = [frame]
        self.cur_frame1 = frame.copy()
        self.buffer_frame_num1 = 0
        self.frame_num1 = 0        
        self.length1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # open second VideoCapture
        self.cap2 = cv2.VideoCapture(sequence2)
        ret,frame2 = self.cap2.read()
        if self.ds != 1:
            frame2 = cv2.resize(frame,(frame2.shape[1]//self.ds,frame2.shape[0]//self.ds))
        
        self.frame_buffer2 = [frame2]
        self.cur_frame2 = frame2.copy()
        self.buffer_frame_num2 = 0
        self.frame_num2 = 0        
        self.length2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        
        
        
        
        self.start_point = None # used to store click temporarily
        self.clicked = False 
        self.new = None # used to store a new box to be plotted temporarily
        self.cont = True
        self.colors = (np.random.rand(100000,3))*255
        
        # load labels
        frame_labels = {}
        with open(self.label_path1,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                
                if len(row) == 0 :
                    continue
                
                if not HEADERS:
                    frame_idx = int(row[0])
                    
                    if frame_idx not in frame_labels.keys():
                        frame_labels[frame_idx] = [row]
                    else:
                        frame_labels[frame_idx].append(row)
                        
                if HEADERS and len(row) > 0:
                    if row[0][0:5] == "Frame":
                        HEADERS = False # all header lines have been read at this point
                        
        
        self.labels1 = frame_labels
    
        
        frame_labels = {}
        with open(self.label_path2,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                
                if len(row) == 0 :
                    continue
                
                if not HEADERS:
                    frame_idx = int(row[0])
                    
                    if frame_idx not in frame_labels.keys():
                        frame_labels[frame_idx] = [row]
                    else:
                        frame_labels[frame_idx].append(row)
                        
                if HEADERS and len(row) > 0:
                    if row[0][0:5] == "Frame":
                        HEADERS = False # all header lines have been read at this point
                        
        
        self.labels2 = frame_labels
        self.label_buffer = [(copy.deepcopy(self.labels1),copy.deepcopy(self.labels2))]
    
    
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
              
                 
    def plot(self,show_lines = True):
        """
        Replots current frame
        """
        try:
            cur_frame_objs = self.labels1[self.frame_num1].copy()
        except:
            cur_frame_objs = []
            
        index_in_buffer = -(self.buffer_frame_num1 - self.frame_num1) - 1
        self.cur_frame1 = self.frame_buffer1[index_in_buffer].copy()
        
        frame_a_boxes = {}
        frame_b_boxes = {}
        
        for box in cur_frame_objs:
            try:
                cls = box[3]
                oidx = int(box[2])
                bbox = np.array(box[4:8]).astype(float).astype(int)
                color = self.colors[oidx%100000]
                frame_a_boxes[oidx] = bbox
                
                label = "{} {}".format(cls,oidx)
                self.cur_frame1 = cv2.rectangle(self.cur_frame1,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,2)
                self.cur_frame1 = cv2.putText(self.cur_frame1,"{}".format(label),(int(bbox[0]),int(bbox[1] - 10)),cv2.FONT_HERSHEY_PLAIN,4,(0,0,0),7)
                self.cur_frame1 = cv2.putText(self.cur_frame1,"{}".format(label),(int(bbox[0]),int(bbox[1] - 10)),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),3)
            except:
                pass
        
        #repeat for second half
        try:
            cur_frame_objs = self.labels2[self.frame_num2].copy()
        except:
            cur_frame_objs = []
            
        index_in_buffer = -(self.buffer_frame_num2 - self.frame_num2) - 1
        self.cur_frame2 = self.frame_buffer2[index_in_buffer].copy()
        
        for box in cur_frame_objs:
            cls = box[3]
            oidx = int(box[2])
            bbox = np.array(box[4:8]).astype(float).astype(int)
            color = self.colors[oidx%100000]
            frame_b_boxes[oidx] = bbox
            
            label = "{} {}".format(cls,oidx)
            self.cur_frame2 = cv2.rectangle(self.cur_frame2,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,2)
            self.cur_frame2 = cv2.putText(self.cur_frame2,"{}".format(label),(int(bbox[0]),int(bbox[1] - 10)),cv2.FONT_HERSHEY_PLAIN,4,(0,0,0),7)
            self.cur_frame2 = cv2.putText(self.cur_frame2,"{}".format(label),(int(bbox[0]),int(bbox[1] - 10)),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),3)
            
        self.cur_frame = np.hstack((self.cur_frame1,self.cur_frame2))
        
        for oidx in frame_a_boxes.keys():
            if oidx in frame_b_boxes.keys():
                
                corner_a = frame_a_boxes[oidx][0:2]
                corner_b = frame_b_boxes[oidx][0:2]
                corner_b[0] += 3840
                color = self.colors[oidx%100000]
                
                self.cur_frame = cv2.line(self.cur_frame,(int(corner_a[0]),int(corner_a[1])),(int(corner_b[0]),int(corner_b[1])),color,2)
                
     
    def associate(self,a_idx,b_idx):
        """
        Reassign all objects a_idx to all objects with b_idx
        """
        
        frame_idx = 0
        cls = None
        
        # get class
        while frame_idx < self.length1 and cls is None:
            try:
                for idx,row in enumerate(self.labels1[frame_idx]):
                    if int(row[2]) == a_idx:
                        cls = row[3]           
            except KeyError:
                pass
            frame_idx += 1
        
        frame_idx = 0
        while frame_idx < self.length2:
            try:
                for idx,row in enumerate(self.labels2[frame_idx]):
                    if int(row[2]) == b_idx:
                        row[2] = a_idx
                        row[3] = cls
            except KeyError:
                pass
            frame_idx += 1
        
        print("Changed obj {} to {} in all frames".format(b_idx,a_idx,))
        
    def reassign_class(self,oidx,cls):
        """
        Assign the new class to all boxes with the obj index
        """
        frame_idx = 0
        cls_old = "Unknown"
        
        while frame_idx < self.length1:
            try:
                for idx,row in enumerate(self.labels1[frame_idx]):
                    if int(row[2]) == oidx:
                        cls_old = row[3]
                        row[3] = cls
            except KeyError:
                pass
            frame_idx += 1
        
        frame_idx = 0
        while frame_idx < self.length2:
            try:
                for idx,row in enumerate(self.labels2[frame_idx]):
                    if int(row[2]) == oidx:
                        cls_old = row[3]
                        row[3] = cls
            except KeyError:
                pass
            frame_idx += 1
        
        print("Changed obj {} class from {} to {} in all frames".format(oidx,cls_old,cls))
        self.plot()
        
    def undo(self):
        """
        undo the last operation for this frame 
        """
        if len(self.label_buffer) > 1:
            
            self.labels1 = self.label_buffer[-2][0].copy()
            self.labels2 = self.label_buffer[-2][1].copy()
            self.label_buffer = self.label_buffer[:-1]
            
            self.plot()
            print("Undid last operation")
        
        else:
            print("Can't undo the last operation")
       
    def find_box(self,point):
        point = point * self.plot_ds
        
        # subtract offset from second x coordinate
        point[2]  = point[2] - 3840
        
        a = -1
        b = -1
        
        for row in self.labels1[self.frame_num1]:
            bbox = np.array(row[4:8]).astype(float).astype(int)
             # since we downsample image before plotting
            if point[0] > bbox[0] and point[0] < bbox[2] and point[1] > bbox[1] and point[1] < bbox[3]:
                a = int(row[2])
                break
        
        for row in self.labels2[self.frame_num2]:
            bbox = np.array(row[4:8]).astype(float).astype(int)
             # since we downsample image before plotting
            if point[2] > bbox[0] and point[2] < bbox[2] and point[3] > bbox[1] and point[3] < bbox[3]:
                b = int(row[2])
                break
        
        # determine whether point falls within any object boxes
        return a,b
        
    def next(self,frame = 1):
        """
        Advance to next frame
        """
        self.clicked = False
        
        if frame ==1:
        
            if self.frame_num1 < self.length1 - 1:
                
                if self.frame_num1 == self.buffer_frame_num1:
                    ret,frame = self.cap1.read()
                    if self.ds != 1:
                        frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
                    self.frame_buffer1.append(frame.copy())
                    self.cur_frame1 = frame.copy()
                    
                    if len(self.frame_buffer1) > 200:
                        self.frame_buffer1 = self.frame_buffer1[1:]
                    
                    self.frame_num1 += 1
                    self.buffer_frame_num1 += 1
                
                    self.plot()
                
                else:
                    # get frame frome frame_buffer
                    self.frame_num1 += 1
                    self.plot()
            
            
            else:
                print("On last frame, cannot advance to next frame")
        
        elif frame == 2:
            if self.frame_num2 < self.length2 - 1:
                
                if self.frame_num2 == self.buffer_frame_num2:
                    ret,frame = self.cap2.read()
                    if self.ds != 1:
                        frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
                    self.frame_buffer2.append(frame.copy())
                    self.cur_frame2 = frame.copy()
                    
                    if len(self.frame_buffer2) > 200:
                        self.frame_buffer2 = self.frame_buffer2[1:]
                    
                    self.frame_num2 += 1
                    self.buffer_frame_num2 += 1
                
                    self.plot()
                
                else:
                    # get frame frome frame_buffer
                    self.frame_num2 += 1
                    self.plot()
       
    def prev(self,frame = 1):
        """
        Return to previous frame, which is stored in a temporary buffer
        """
        self.clicked = False
        
        if frame == 1:
            if self.frame_num1 > 0:
                self.frame_num1 -= 1
                self.plot()
            else:
                print("On first frame, cannot return to previous frame")
                
        elif frame == 2:
            if self.frame_num2 > 0:
                self.frame_num2 -= 1  
                self.plot()
            
            else:
                print("On first frame, cannot return to previous frame")
    
    def save(self):
        """
        Saves the current annotations to CSV 
        """
        
        # load csv file annotations into list
        output_rows = []
        with open(self.label_path1,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                
                if HEADERS:
                    if len(row) > 0 and row[0][0:5] == "Frame":
                        HEADERS = False # all header lines have been read at this point
                        output_rows.append(row)
                        break
                else:
                    output_rows.append(row)

        frames = list(self.labels1.keys())
        frames.sort()
        
        for frame in frames:
            for row in self.labels1[frame]:
                output_rows.append(row)
    
        
        with open(self.label_path1, mode='w') as f:
            out = csv.writer(f, delimiter=',')
            out.writerows(output_rows)
            
            
        # repeat for file 2
            
        # load csv file annotations into list
        output_rows = []
        with open(self.label_path2,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                
                if HEADERS:
                    if len(row) > 0 and row[0][0:5] == "Frame":
                        HEADERS = False # all header lines have been read at this point
                        output_rows.append(row)
                        break
                else:
                    output_rows.append(row)

        frames = list(self.labels2.keys())
        frames.sort()
        
        for frame in frames:
            for row in self.labels2[frame]:
                output_rows.append(row)
    
        
        with open(self.label_path2, mode='w') as f:
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
        
        changes = 0
        while(self.cont): # one frame
        
          
           # handle click actions
           if self.new is not None:
               
                # check whether box starts and ends within an object box
                a,b = self.find_box(self.new)

                # if so, associate these objects
                if a != -1 and b != -1:
                    self.associate(a,b)
                
                elif a != -1:
                    cls = self.keyboard_input()
                    self.reassign_class(a,cls)
                elif b != -1:
                    cls = self.keyboard_input()
                    self.reassign_class(b,cls)
                
                # update label buffer
                self.label_buffer.append((copy.deepcopy(self.labels1),copy.deepcopy(self.labels2)))
                if len(self.label_buffer) > 50:
                    self.label_buffer = self.label_buffer[1:]
                 
                # update plotted frames
                self.plot()
                self.new = None     
                
                changes += 1
                if changes == 20:
                    changes = 0
                    self.save()
           
             
           cur_frame = self.cur_frame.copy()
           #cur_frame = cv2.resize(cur_frame,(cur_frame.shape[1]//self.plot_ds,cur_frame.shape[0]//self.plot_ds))
           cv2.imshow("window", cur_frame)
           title = "Frame {}/{} of {}/{}".format(self.frame_num1,self.length1,self.frame_num2,self.length2)
           cv2.setWindowTitle("window",str(title))
           
           key = cv2.waitKey(1)
           if   key == ord('2'):
                self.next(frame = 1)
           elif key == ord('1'):
                self.prev(frame = 1)
           elif key == ord('3'):
                self.prev(frame = 2)     
           elif key == ord('4'):
                self.next(frame = 2)     
           elif key == ord("q"):
                self.quit()
           elif key == ord("u"):
                self.undo()

        
        
if __name__ == "__main__":
    
    # sequence1 = "/home/worklab/Data/cv/video/5_min_18_cam_October_2020/ingest_session_00005/recording/record_p2c6_00001.mp4"
    # sequence2 = "/home/worklab/Data/cv/video/5_min_18_cam_October_2020/ingest_session_00005/recording/record_p2c5_00001.mp4"
    # label_dir = "/home/worklab/Documents/derek/i24-dataset-gen/output"
    
    sequence1 = "/home/worklab/Data/cv/video/08_06_2021/record_51_p1c2_00000.mp4"
    sequence2 = "/home/worklab/Data/cv/video/08_06_2021/record_51_p1c3_00000.mp4"
    label_dir = "/home/worklab/Data/cv/video/08_06_2021/rectified"
    
    
    #test_path = "C:\\Users\\derek\\Desktop\\2D to 3D conversion Examples April 2021-selected\\record_p1c2_00000.mp4"
    ann = Annotator_2D(sequence1,sequence2,label_dir)
    ann.run()