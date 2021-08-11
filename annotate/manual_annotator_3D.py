# using openCV, draw ignored regions for each frame. 

import numpy as np
import os
import cv2
import csv
import copy
import argparse
import string
import cv2 as cv

from vp_utils import find_vanishing_point

class Annotator_3D():
    
    def __init__(self,sequence,label_file, transform_point_file,vp_file, ds = 1,load_corrected = False):
        """
        sequence - path to video file
        """
        
        self.sequence_path = sequence
        self.sequence_short_name = sequence.split("/")[-1].split(".mp4")[0]
        self.camera_name = self.sequence_short_name.split("_")[0]
        
        self.label_path = label_file
        self.load_corrected = False
        
        
        self.outfile = label_file # self.label_path.split(".csv")[0] + "_corrected.csv"
        print (self.outfile)
        if load_corrected:
            if os.path.exists(self.outfile):
                self.label_path = self.outfile
                self.load_corrected = True
        
        # write final output file
        
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
        self.keyframe_point = None
        self.clicked = False 
        self.new = None # used to store a new box to be plotted temporarily
        self.cont = True
        self.active_command = None
        self.active_vp = 0
        
        self.colors = (np.random.rand(100000,3))*255
        self.last_active_obj_idx = -1

        self.changes_since_last_save = 0

        # get transform from RWS -> ImS
        keep = []
        with open(transform_point_file,"r") as f:
            read = csv.reader(f)
            FIRST = True
            for row in read:
                if FIRST:
                    FIRST = False
                    continue
                        
                if "im space"  in row[0]:
                    break
                
                keep.append(row)
        pts = np.stack([[float(item) for item in row] for row in keep])
        im_pts = pts[:,:2]
        lmcs_pts = pts[:,2:]
        self.H,_ = cv2.findHomography(lmcs_pts,im_pts)
        
        # get vanishing points
        lines1 = []
        lines2 = []
        lines3 = []
        with open(vp_file,"r") as f:
            read = csv.reader(f)
            for item in read:
                if item[4] == '0':
                    lines1.append(np.array(item).astype(float))
                elif item[4] == '1':
                    lines2.append(np.array(item).astype(float))
                elif item[4] == '2':
                    lines3.append(np.array(item).astype(float))
        
        # get all axis labels for a particular axis orientation
        vp1 = find_vanishing_point(lines1)
        vp2 = find_vanishing_point(lines2)
        vp3 = find_vanishing_point(lines3)
        self.vps = [vp1,vp2,vp3]
        
        
        # load labels
        frame_labels = {}
        self.known_heights = {}
        self.known_classes = {}
        
        with open(self.label_path,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                if HEADERS:
                    HEADERS = False
                    continue
                
                frame_idx = int(row[0])
                camera = row[36]
                if camera != self.camera_name:
                    continue
                
                
                if row[10] == "Manual":
                    if frame_idx not in frame_labels.keys():
                            frame_labels[frame_idx] = [row]
                    else:
                        frame_labels[frame_idx].append(row)
                    continue
                            
                if len(row[39]) == 0: # bad data row
                    #print("bad data row")
                    continue
                
                # add image space box points
                x_center = float(row[39]) * 3.281
                y_center = float(row[40]) * 3.281
                theta = float(row[41])
                width = float(row[42])    * 3.281
                length = float(row[43])   * 3.281
                
                fx = x_center + length * np.cos(theta) 
                ly = y_center + width/2 * np.cos(theta)
                bx = x_center
                ry = y_center - width/2 * np.cos(theta)
                
                lmcs_bbox = np.array([[fx,fx,bx,bx],[ry,ly,ry,ly],[1,1,1,1]])#.transpose()
                
                out = np.matmul(self.H,lmcs_bbox)
                im_footprint = np.zeros([2,4])
                im_footprint[0,:] = out[0,:] / out[2,:]
                im_footprint[1,:] = out[1,:] / out[2,:]
                im_footprint = im_footprint.transpose()
                
                
                obj_id = int(row[2])
                if obj_id not in self.known_classes.keys():
                    self.known_classes[obj_id] = row[3]
                
                if obj_id in self.known_heights.keys() and self.known_heights[obj_id] < 200:
                    est_height = self.known_heights[obj_id]
                else:
                    # estimate height
                    try:
                        old_im_pts = np.array(row[11:27]).astype(float)
                        est_height = (np.sum(old_im_pts[[1,3,5,7]]) - np.sum(old_im_pts[[9,11,13,15]]))/4.0
                        self.known_heights[obj_id] = est_height
                    except:
                        est_height = 1
                        
                im_top = im_footprint.copy()
                im_top[:,1] -= est_height
                
                im_bbox = list(im_footprint.reshape(-1)) + list(im_top.reshape(-1)) 
                row[11:27] = im_bbox
                
                # verify that object is within frame
                xmax = max(im_bbox[::2])
                ymax = max(im_bbox[1::2])
                xmin = min(im_bbox[::2])
                ymin = min(im_bbox[1::2])
                if ymax > 0 and xmax > 0 and xmin < 1980 and ymin < 1080:
                    if frame_idx not in frame_labels.keys():
                        frame_labels[frame_idx] = [row]
                    else:
                        frame_labels[frame_idx].append(row)
                        
                ### Temp early cutoff for loading speed
                # if frame_idx > 600:
                #     break
                
                
        self.labels = frame_labels
        self.label_buffer = [[self.frame_num,copy.deepcopy(frame_labels[self.frame_num])]]
    
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
              
       elif event == cv.EVENT_RBUTTONDOWN:
            obj_idx = self.find_box((x,y))
            self.realign(obj_idx, self.frame_num)
            self.plot()
            
    def plot(self):
        """
        Replots current frame
        """
        try:
            cur_frame_objs = self.labels[self.frame_num].copy()
        except:
            cur_frame_objs = []
            
        index_in_buffer = -(self.buffer_frame_num - self.frame_num) - 1
        self.cur_frame = self.frame_buffer[index_in_buffer].copy()
        
        for box in cur_frame_objs:
            oidx = int(box[2])
            cls = box[3]
            if cls == "":
                cls = self.known_classes[oidx]
            bbox = np.array(box[11:27]).astype(float).astype(int) 
            color = self.colors[oidx%100000]
            ts = 2
            self.cur_frame = cv2.line(self.cur_frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,ts)
            self.cur_frame = cv2.line(self.cur_frame,(bbox[6],bbox[7]),(bbox[2],bbox[3]),(0,255,0),2)
            self.cur_frame = cv2.line(self.cur_frame,(bbox[0],bbox[1]),(bbox[4],bbox[5]),color,ts)
            self.cur_frame = cv2.line(self.cur_frame,(bbox[6],bbox[7]),(bbox[4],bbox[5]),(255,0,0),2)
            
            self.cur_frame = cv2.line(self.cur_frame,(bbox[0],bbox[1]),(bbox[8],bbox[9]),color,ts)
            self.cur_frame = cv2.line(self.cur_frame,(bbox[2],bbox[3]),(bbox[10],bbox[11]),color,ts)
            self.cur_frame = cv2.line(self.cur_frame,(bbox[4],bbox[5]),(bbox[12],bbox[13]),color,ts)
            self.cur_frame = cv2.line(self.cur_frame,(bbox[6],bbox[7]),(bbox[14],bbox[15]),(0,0,255),2)
            
            self.cur_frame = cv2.line(self.cur_frame,(bbox[10],bbox[11]),(bbox[8],bbox[9]),color,ts)
            self.cur_frame = cv2.line(self.cur_frame,(bbox[12],bbox[13]),(bbox[8],bbox[9]),color,ts)
            self.cur_frame = cv2.line(self.cur_frame,(bbox[14],bbox[15]),(bbox[12],bbox[13]),color,ts)
            self.cur_frame = cv2.line(self.cur_frame,(bbox[10],bbox[11]),(bbox[14],bbox[15]),color,ts)
            
            # x across back of vehicle            
            # self.cur_frame = cv2.line(self.cur_frame,(bbox[6],bbox[7]),(bbox[12],bbox[13]),color,ts)
            # self.cur_frame = cv2.line(self.cur_frame,(bbox[4],bbox[5]),(bbox[14],bbox[15]),color,ts)

            
            minx = np.min(bbox[::2])
            miny = np.min(bbox[1::2])
            label = "{} {}".format(cls,oidx)
            self.cur_frame = cv2.putText(self.cur_frame,"{}".format(label),(int(minx),int(miny - 10)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),3)
            self.cur_frame = cv2.putText(self.cur_frame,"{}".format(label),(int(minx),int(miny - 10)),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        
            # vanishing points
            # self.cur_frame = cv2.line(self.cur_frame,(bbox[0],bbox[1]),(int(self.vps[0][0]),int(self.vps[0][1])),color,1)
            # self.cur_frame = cv2.line(self.cur_frame,(bbox[0],bbox[1]),(int(self.vps[1][0]),int(self.vps[1][1])),color,1)
            # self.cur_frame = cv2.line(self.cur_frame,(bbox[0],bbox[1]),(int(self.vps[2][0]),int(self.vps[2][1])),color,1)
        
        # label = "Remove all boxes not on a vehicle"
        # self.cur_frame = cv2.putText(self.cur_frame,"{}".format(label),(10,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
        # label = "Add boxes for all objects fully in frame"
        # self.cur_frame = cv2.putText(self.cur_frame,"{}".format(label),(10,80),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
        # label = "Ensure each object has only one ID"
        # self.cur_frame = cv2.putText(self.cur_frame,"{}".format(label),(10,110),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)

        self.changes_since_last_save += 1
        if self.changes_since_last_save > 10:
                self.save()
                self.changes_since_last_save = 0


    def delete(self,obj_idx, n_frames = -1):
        """
        Delete object obj_idx in this and n_frames -1 subsequent frames. If n_frames 
        = -1, deletes obj_idx in all subsequent frames
        """
        frame_idx = self.frame_num 
        
        stop_idx = self.frame_num + n_frames 
        if n_frames == -1:
            stop_idx = self.length
        
        while frame_idx < stop_idx:
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
                    break
            if cls is not None:
                break
        
        if cls is None: # try again but accepting boxes with no speed estimate
            for frame in self.labels:
                labels = self.labels[frame]
                for box in labels:
                    if int(box[2]) == obj_idx:
                        cls = box[3]
                        break
                if cls is not None:
                    break
            
        if cls is None:
            print("This object does not exist elsewhere. Input class:")
            cls = self.keyboard_input()
        
        timestamp = ""
        try:
            for box in self.labels[self.frame_num]:
                if len(box[1]) > 0:
                    timestamp = box[1]
                    break
        except:
            pass
        
        size = 100
        newbox = np.array([bbox[0],bbox[1],bbox[0] + size,bbox[1],bbox[0],bbox[1] + size, bbox[0]+size,bbox[1] + size])
        newbox2 = newbox + size/2.0
        new_row = [self.frame_num,timestamp,obj_idx,cls] + ["","","","","","",""] + list(newbox2) + list(newbox) + ["" for tmp in range(9)] + [self.camera_name] + ["" for tmp in range(7)]
        try:
            self.labels[self.frame_num].append(new_row)
        except:
            self.labels[self.frame_num] = [new_row]
        
        print("Added box for object {} to frame {}".format(obj_idx,self.frame_num))
        self.realign(obj_idx,self.frame_num)
        #self.next() # for convenience
        
    def move(self,obj_idx,disp):
        
        x_move = disp[2] - disp[0]
        y_move = disp[3] - disp[1]
        
        for row in self.labels[self.frame_num]:
            if int(row[2]) == obj_idx:
                bbox = np.array(row[11:27]).astype(float)
                
                bbox[0] += x_move
                bbox[2] += x_move
                bbox[4] += x_move
                bbox[6] += x_move
                bbox[8] += x_move
                bbox[10] += x_move
                bbox[12] += x_move
                bbox[14] += x_move
                bbox[1] += y_move
                bbox[3] += y_move
                bbox[5] += y_move
                bbox[7] += y_move
                bbox[9] += y_move
                bbox[11] += y_move
                bbox[13] += y_move
                bbox[15] += y_move
                
                row[10] = "Manual"
                row[11:27]  = bbox.astype(int)
                row[27:] = ["" for i in range(16)]
                row[36] = self.camera_name
                
                break
    
    
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
     
    def realign(self,obj_idx,target_frame_num):
        """
        Redraws selected box [obj_idx] in target_frame_num so that it is 
        axis-aligned according to vanishing points
        
        There's probably a better way to do this but:
            1. draw line from vp1 to fbr 
            2. shift bbr to fall on this line
            3. draw line from vp2 to bbr
            4. shift bbl to fall on this line
            5. draw lines through bbl and fbr and find intersection - move fbl to this point
            6. draw line and align ftr
            7. draw lines two lines and intersect btr
            8. btl
            9. ftl
        """
        
        for row in self.labels[target_frame_num]:
            if int(row[2]) == obj_idx:
                bbox = np.array(row[11:27]).astype(float)
                
                # 1. - realign bbr
                # express bbr as a vector from fbr (mag,dir)
                mag = np.sqrt((bbox[4] - bbox[0])**2 + (bbox[5] - bbox[1])**2)
                
                # get angle to vp
                dr  = np.arctan((self.vps[0][1] - bbox[1])/(self.vps[0][0] - bbox[0]))
                if (self.vps[0][0] - bbox[0]) < 0:
                    dr = dr + np.pi
                
                # get sign term that is + if vector points towards vp, - otherwise
                vp_length = np.sqrt((self.vps[0][0] - bbox[0])**2 + (self.vps[0][1] - bbox[1])**2)
                perp_comp = ((self.vps[0][0]-bbox[0]) * (bbox[4] - bbox[0]) + (self.vps[0][1]-bbox[1]) * (bbox[5] - bbox[1]))/(vp_length*mag)
                sign = np.sign(perp_comp)
                
                # realign
                bbox[4] = bbox[0] + sign* np.cos(dr) * mag
                bbox[5] = bbox[1] + sign* np.sin(dr) * mag
                
                
                #2 - realign bbl
                # express bbl as a vector from bbr (mag,dir)
                mag = np.sqrt((bbox[6] - bbox[4])**2 + (bbox[7] - bbox[5])**2)
                
                # get angle to vp
                dr  = np.arctan((self.vps[1][1] - bbox[5])/(self.vps[1][0] - bbox[4]))
                if (self.vps[1][0] - bbox[4]) < 0:
                    dr = dr + np.pi
                
                vp_length = np.sqrt((self.vps[1][0] - bbox[4])**2 + (self.vps[1][1] - bbox[5])**2)
                perp_comp = ((self.vps[1][0]-bbox[4]) * (bbox[6] - bbox[4]) + (self.vps[1][1]-bbox[5]) * (bbox[7] - bbox[5]))/(vp_length*mag)
                sign = np.sign(perp_comp)
                
                # realign
                bbox[6] = bbox[4] + sign* np.cos(dr) * mag
                bbox[7] = bbox[5] + sign* np.sin(dr) * mag
                
                
                
                # 3. realign fbl
                # find intersection of lines
                x1,y1,x2,y2 = bbox[6],bbox[7],self.vps[0][0],self.vps[0][1]
                x3,y3,x4,y4 = bbox[0],bbox[1],self.vps[1][0],self.vps[1][1]
                
                D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                bbox[2] = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-x4*y3))/D
                bbox[3] = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-x4*y3))/D
                

                # 4. realign ftr
                # express ftr as a vector from fbr (mag,dir)
                mag = np.sqrt((bbox[8] - bbox[0])**2 + (bbox[9] - bbox[1])**2)
                
                # get angle to vp
                dr  = np.arctan((self.vps[2][1] - bbox[1])/(self.vps[2][0] - bbox[0]))
                if (self.vps[2][0] - bbox[0]) < 0:
                    dr = dr + np.pi
                
                # get sign term that is + if vector points towards vp, - otherwise
                vp_length = np.sqrt((self.vps[2][0] - bbox[0])**2 + (self.vps[2][1] - bbox[1])**2)
                perp_comp = ((self.vps[2][0]-bbox[0]) * (bbox[8] - bbox[0]) + (self.vps[2][1]-bbox[1]) * (bbox[9] - bbox[1]))/(vp_length*mag)
                sign = np.sign(perp_comp)
                
                # realign
                bbox[8] = bbox[0] + sign* np.cos(dr) * mag
                bbox[9] = bbox[1] + sign* np.sin(dr) * mag
                
                
                # 5. realign btr
                # find intersection of lines
                x1,y1,x2,y2 = bbox[8],bbox[9],self.vps[0][0],self.vps[0][1]
                x3,y3,x4,y4 = bbox[4],bbox[5],self.vps[2][0],self.vps[2][1]
                
                D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                bbox[12] = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-x4*y3))/D
                bbox[13] = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-x4*y3))/D
                
                #6. realign btl
                x1,y1,x2,y2 = bbox[12],bbox[13],self.vps[1][0],self.vps[1][1]
                x3,y3,x4,y4 = bbox[6],bbox[7],self.vps[2][0],self.vps[2][1]
                
                D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                bbox[14] = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-x4*y3))/D
                bbox[15] = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-x4*y3))/D
                
                #6. realign ftl
                x1,y1,x2,y2 = bbox[14],bbox[15],self.vps[0][0],self.vps[0][1]
                x3,y3,x4,y4 = bbox[2],bbox[3],self.vps[2][0],self.vps[2][1]
                
                D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                bbox[10] = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-x4*y3))/D
                bbox[11] = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-x4*y3))/D
                
                
                row[11:27] = bbox.astype(int)
                row[10] = "Manual"
                row[27:] = ["" for i in range(16)]
                row[36] = self.camera_name
                break
            
            print("Realigned object {} in frame {}".format(obj_idx,target_frame_num))
        
    
    def redraw(self,obj_idx,disp):
        """
        Attention recruiters - plz don't judge me too harshly for this function.
        I know a lot of the code could have been condensed rather than repeated, 
        and frankly I'm upset about it too. But sometimes you have to sacrifice 
        beauty for utility and honestly it would've taken me a few extra hours to
        do it right so I hope you can respect that.
        """
        
        if disp[0] == disp[2] and disp[1] == disp[3]:
            return
        
        for row in self.labels[self.frame_num]:
            if int(row[2]) == obj_idx:
                bbox = np.array(row[11:27]).astype(float)
                
                # determine whether start point was closer to one side or the other of the box based on active dim
                if self.active_vp == 0:
                    x1 = (bbox[0] + bbox[2] + bbox[8]  + bbox[10])/4.0
                    x2 = (bbox[4] + bbox[6] + bbox[12] + bbox[14])/4.0
                    y1 = (bbox[1] + bbox[3] + bbox[9] + bbox[11])/4.0
                    y2 = (bbox[5] + bbox[7] + bbox[13] + bbox[15])/4.0
                    
                    d1 = (x1 - disp[0])**2 + (y1 - disp[1])**2
                    d2 = (x2 - disp[0])**2 + (y2 - disp[1])**2
                    
                    lx = x1 - x2
                    ly = y1 - y2
                    length = np.sqrt(lx**2 + ly**2)
                    disp_length = np.sqrt((disp[2] -disp[0])**2 + (disp[3] - disp[1])**2)
                    
                    perp_comp = ((disp[2]-disp[0]) * lx + (disp[3] - disp[1])*ly)/(disp_length*length)
                    disp_length *= perp_comp
                    
                    
                    if d1 < d2:
                        # move front
                        lx *= (length+disp_length)/length
                        ly *= (length+disp_length)/length
                        
                        
                        bbox[0] = bbox[4] + lx 
                        bbox[2] = bbox[6] + lx 
                        bbox[8] = bbox[12] + lx 
                        bbox[10] = bbox[14] + lx 
                        bbox[1] = bbox[5] + ly 
                        bbox[3] = bbox[7] + ly 
                        bbox[9] = bbox[13] + ly 
                        bbox[11] = bbox[15] + ly 
                    else:
                        
                        lx *= (-length+disp_length)/length
                        ly *= (-length+disp_length)/length
                        
                        bbox[4] = bbox[0] + lx 
                        bbox[6] = bbox[2] + lx 
                        bbox[12] = bbox[8] + lx 
                        bbox[14] = bbox[10] + lx 
                        bbox[5] = bbox[1] + ly 
                        bbox[7] = bbox[3] + ly
                        bbox[13] = bbox[9] + ly 
                        bbox[15] = bbox[11] + ly 

                    row[11:27] = bbox
                    row[10] = "Manual"
                    row[27:] = ["" for i in range(16)]
                    row[36] = self.camera_name

                
                elif self.active_vp == 1:
                    
                
                    x1 = (bbox[0] + bbox[4] + bbox[8]  + bbox[12])/4.0 # RIGHT SIDE
                    x2 = (bbox[2] + bbox[6] + bbox[10] + bbox[14])/4.0 # LEFT SIDE
                    y1 = (bbox[1] + bbox[5] + bbox[9] + bbox[13])/4.0
                    y2 = (bbox[3] + bbox[7] + bbox[11] + bbox[15])/4.0
        
                    d1 = (x1 - disp[0])**2 + (y1 - disp[1])**2
                    d2 = (x2 - disp[0])**2 + (y2 - disp[1])**2
                    
                    lx = x1 - x2
                    ly = y1 - y2
                    length = np.sqrt(lx**2 + ly**2)
                    disp_length = np.sqrt((disp[2] -disp[0])**2 + (disp[3] - disp[1])**2)
                    
                    perp_comp = ((disp[2]-disp[0]) * lx + (disp[3] - disp[1])*ly)/(disp_length*length)
                    disp_length *= perp_comp
                    
                    
                    if d1 < d2:
                        # move front
                        lx *= (length+disp_length)/length
                        ly *= (length+disp_length)/length
                        
                        
                        bbox[0] = bbox[2] + lx 
                        bbox[4] = bbox[6] + lx 
                        bbox[8] = bbox[10] + lx 
                        bbox[12] = bbox[14] + lx 
                        bbox[1] = bbox[3] + ly 
                        bbox[5] = bbox[7] + ly 
                        bbox[9] = bbox[11] + ly 
                        bbox[13] = bbox[15] + ly 
                    else:
                        
                        lx *= (-length+disp_length)/length
                        ly *= (-length+disp_length)/length
                        
                        bbox[2] = bbox[0] + lx 
                        bbox[6] = bbox[4] + lx 
                        bbox[10] = bbox[8] + lx 
                        bbox[14] = bbox[12] + lx 
                        bbox[3] = bbox[1] + ly 
                        bbox[7] = bbox[5] + ly
                        bbox[11] = bbox[9] + ly 
                        bbox[15] = bbox[13] + ly 
                        
                    row[11:27] = bbox
                    row[10] = "Manual"
                    row[27:] = ["" for i in range(16)]
                    row[36] = self.camera_name
                    
                elif self.active_vp == 2:
                    
                
                    x1 = (bbox[0] + bbox[4] + bbox[6]  + bbox[2])/4.0 # BOTTOM SIDE
                    x2 = (bbox[8] + bbox[12] + bbox[10] + bbox[14])/4.0 # TOP SIDE
                    y1 = (bbox[1] + bbox[3] + bbox[5] + bbox[7])/4.0
                    y2 = (bbox[9] + bbox[11] + bbox[13] + bbox[15])/4.0
        
                    d1 = (x1 - disp[0])**2 + (y1 - disp[1])**2
                    d2 = (x2 - disp[0])**2 + (y2 - disp[1])**2
                    
                    lx = x1 - x2
                    ly = y1 - y2
                    length = np.sqrt(lx**2 + ly**2)
                    disp_length = np.sqrt((disp[2] -disp[0])**2 + (disp[3] - disp[1])**2)
                    
                    perp_comp = ((disp[2]-disp[0]) * lx + (disp[3] - disp[1])*ly)/(disp_length*length)
                    disp_length *= perp_comp
                    
                    
                    if d1 < d2:
                        # move front
                        lx *= (length+disp_length)/length
                        ly *= (length+disp_length)/length
                        
                        
                        bbox[0] = bbox[8] + lx 
                        bbox[2] = bbox[10] + lx 
                        bbox[4] = bbox[12] + lx 
                        bbox[6] = bbox[14] + lx 
                        bbox[1] = bbox[9] + ly 
                        bbox[3] = bbox[11] + ly 
                        bbox[5] = bbox[13] + ly 
                        bbox[7] = bbox[15] + ly 
                    else:
                        
                        lx *= (-length+disp_length)/length
                        ly *= (-length+disp_length)/length
                        
                        bbox[8]  = bbox[0] + lx 
                        bbox[10] = bbox[2] + lx 
                        bbox[12] = bbox[4] + lx 
                        bbox[14] = bbox[6] + lx 
                        bbox[9]  = bbox[1] + ly 
                        bbox[11] = bbox[3] + ly
                        bbox[13] = bbox[5] + ly 
                        bbox[15] = bbox[7] + ly 
                        
                    row[11:27] = bbox
                    row[10] = "Manual"
                    row[27:] = ["" for i in range(16)]
                    row[36] = self.camera_name
                break
         
        #self.realign(obj_idx, self.frame_num)
        print("Redrew box {} for frame {}".format(obj_idx,self.frame_num))
        
    
    def undo(self):
        """
        undo the last operation for this frame 
        """
        if len(self.label_buffer) > 1:
            
            prev = self.label_buffer[-2].copy()
            if prev[0] == self.frame_num:
                self.labels[self.frame_num] = prev[1]
                self.label_buffer = self.label_buffer[:-1]
                print("Undid last operation")
                self.plot()
            else:
                print("Cannot undo last operation because you advanced frames")
            
        else:
            print("Can't undo the last operation")
       
    def interpolate(self,obj_idx):
        """
        Interpolate boxes between any gaps in the selected box
        """
        
        prev_frame = -1
        prev_box = None
        cls = None
        for frame in range(0,self.length):
            try:
                for row in self.labels[frame]:
                    if int(row[2]) == obj_idx: # see if obj_idx is in this frame
                        # if we have prev_box, interpolate all boxes between
                        if prev_frame != -1:
                            box = np.array(row[11:27]).astype(float)
                            for f_idx in range(prev_frame + 1, frame):   # for each frame between:
                                p1 = float(frame - f_idx) / float(frame - prev_frame)
                                p2 = 1.0 - p1
                                
                                newbox = p1 * prev_box + p2 * box
                                new_row = [f_idx,"",obj_idx,cls,"","","","","","",""] + list(newbox) + ["" for i in range(16)]
                                new_row[10] = "Manual"
                                new_row[36] = self.camera_name
                                
                                try:
                                    self.labels[f_idx].append(new_row)
                                except:
                                    self.labels[f_idx] = [new_row]
                        
                        # lastly, update prev_frame
                        prev_frame = frame
                        prev_box = np.array(row[11:27]).astype(float)
                        cls = row[3]
                        continue
                    
                    
            except KeyError:
                continue
            
        print("Added interpolated boxes for object {}".format(obj_idx))
        self.plot()
            
    def keyframe(self,obj_idx,point):
        if self.keyframe_point is None:
            
            for row in self.labels[self.frame_num]:
                    if int(row[2]) == obj_idx: # see if obj_idx is in this frame            
                        self.keyframe_point = [obj_idx,point[0:2],np.array(row[11:27]).astype(float)] # store obj_idx, base point, and base box
                        print("Assigned keframe base point for obj {}".format(obj_idx))
                        break
        
        else:
            # calculate offset between point and base point
            x_off = point[0] - self.keyframe_point[1][0]
            y_off = point[1] - self.keyframe_point[1][1]
            
            obj_idx = self.keyframe_point[0]
            
            offset_box = self.keyframe_point[2].copy()
            offset_box[::2]  += x_off
            offset_box[1::2] += y_off
            
            # add dummy box if necessary
            exists = False
            for row in self.labels[self.frame_num]:
                if int(row[2]) == obj_idx:
                    exists = True
                    break
            if not exists:
                self.add(obj_idx,point)
                
            for r_idx,row in enumerate(self.labels[self.frame_num]):
                if int(row[2]) == obj_idx: # see if obj_idx is in this frame   
                    row[11:27] = offset_box
                    row[10] = "Manual"
                    row[27:] = ["" for i in range(16)]
                    row[36] = self.camera_name
                
                    #self.labels[self.frame_num][r_idx] = row
                    print("Used keyframe offset for obj {}".format(obj_idx))
                    break
           
            
            self.realign(obj_idx, self.frame_num)
            self.plot()

        
    def find_box(self,point):
        #point = point *2
        #print(point)

        for row in self.labels[self.frame_num]:
            bbox = np.array(row[11:27]).astype(float).astype(int)
            minx = np.min(bbox[::2])
            maxx = np.max(bbox[::2])
            miny = np.min(bbox[1::2])
            maxy = np.max(bbox[1::2])
             # since we downsample image before plotting
            
            if point[0] > minx and point[0] < maxx and point[1] > miny and point[1] < maxy:
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
                
                if len(self.frame_buffer) > 200:
                    self.frame_buffer = self.frame_buffer[1:]
                
                self.frame_num += 1
                self.buffer_frame_num += 1
            
                self.plot()
            
            else:
                # get frame frome frame_buffer
                self.frame_num += 1
                self.plot()
            
            self.label_buffer = [[self.frame_num,copy.deepcopy(self.labels[self.frame_num])]]

            
        else:
            print("On last frame, cannot advance to next frame")    
    
    def skip_to(self,frame_idx):
         while self.frame_num < frame_idx:
             ret = self.cap.grab()
             self.frame_num += 1
             
         ret,frame = self.cap.retrieve()
         if self.ds != 1:
             frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
         self.frame_buffer.append(frame.copy())
         self.cur_frame = frame.copy()
         self.buffer_frame_num = self.frame_num
         
         self.plot()
        
    def prev(self):
        """
        Return to previous frame, which is stored in a temporary buffer
        """
        self.clicked = False
        if self.frame_num > 0:
            self.frame_num -= 1
            
            self.plot()
            
            self.label_buffer = [[self.frame_num,copy.deepcopy(self.labels[self.frame_num])]]

            
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
                        output_rows.append(row)
                        break
                else:
                    pass#output_rows.append(row)

        frames = list(self.labels.keys())
        frames.sort()
        
        for frame in frames:
            for row in self.labels[frame]:
                output_rows.append(row)
    
        
        with open(self.outfile, mode='w') as f:
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
    
    def scan(self):
        while cv2.waitKey(1) not in [ord("\n"), ord("\r"),ord("f")]:
            self.next()
            
            self.cur_frame = cv2.resize(self.cur_frame,(1920,1080))
            cv2.imshow("window", self.cur_frame)
            title = "Frame {}/{} --- Active Command: {}".format(self.frame_num,self.length,self.active_command)
            cv2.setWindowTitle("window",str(title))
          
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
                    try:
                        n_frames = int(self.keyboard_input())
                    except:
                        n_frames = -1
                        
                    self.delete(obj_idx,n_frames = n_frames)
                    
                elif self.active_command == "ADD":
                    # get obj_idx
                    try:
                        obj_idx = int(self.keyboard_input())
                        self.last_active_obj_idx = obj_idx
                    except:
                        obj_idx = self.last_active_obj_idx
                    
                    
                    #self.new *= 2
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
                    obj_idx = self.find_box(self.new)
                    #self.new *= 2
                    self.redraw(obj_idx,self.new)
                    
                elif self.active_command == "MOVE":
                    obj_idx = self.find_box(self.new)
                    #self.new *= 2
                    self.move(obj_idx,self.new)
                
                elif self.active_command == "REALIGN":
                    obj_idx = self.find_box(self.new)
                    #self.new *= 2
                    self.realign(obj_idx,self.frame_num)
                    
                elif self.active_command == "KEYFRAME":
                    obj_idx = self.find_box(self.new)
                    #self.new *= 2
                    self.keyframe(obj_idx,self.new)
                  
                self.label_buffer.append([self.frame_num,copy.deepcopy(self.labels[self.frame_num])])
                if len(self.label_buffer) > 50:
                    self.label_buffer = self.label_buffer[1:]
                    
                self.plot()
                self.new = None     
           
               
           #self.cur_frame = cv2.resize(self.cur_frame,(1920,1080))
           cv2.imshow("window", self.cur_frame)
           title = "Frame {}/{} --- Active Command: {} --- VP {} ".format(self.frame_num,self.length,self.active_command,self.active_vp)
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
           elif key == ord("l"):
                self.active_command = "REALIGN"
           elif key == ord("m"):
                self.active_command = "MOVE"
           elif key == ord("k"):
               self.keyframe_point = None
               self.active_command = "KEYFRAME"
                
           elif key == ord("s"):
                    frame_idx = int(self.keyboard_input())
                    self.skip_to(frame_idx)          
           elif key == ord("i"):
                    obj_idx = int(self.keyboard_input())
                    self.interpolate(obj_idx)         
           elif key == ord("q"):
                self.quit()
           elif key == ord("u"):
                self.undo()
           elif key == ord("f"):
               self.scan()
           elif key == ord("1"):
               self.active_vp = (self.active_vp +1)%3
           
        
        
if __name__ == "__main__":
    
    try:

        parser = argparse.ArgumentParser()
        parser.add_argument("camera",help = "p1c1 or similar")
        parser.add_argument("index" ,type = int,help = "int in range 0..3")

        args = parser.parse_args()
        sequence_idx = args.index
        camera_id = args.camera

       
        
    except:
        camera_id = "p1c3"
        sequence_idx = 0
        
        
        # label_file = "/home/worklab/Data/dataset_alpha/rectified/rectified_{}_{}.csv".format(camera_id,sequence_idx)
        # video      = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments/{}_{}.mp4".format(camera_id,sequence_idx)
        # vp_file    = "/home/worklab/Data/dataset_alpha/vp/{}_axes.csv".format(camera_id)
        # tf_file    = "/home/worklab/Data/dataset_alpha/tform/{}_im_lmcs_transform_points.csv".format(camera_id)
    
    d = os.path.dirname(os.getcwd())
    label_file = "{}/DATA/labels/rectified_{}_{}_track_outputs_3D.csv".format(d,camera_id,sequence_idx)
    video      = "{}/DATA/video/{}_{}.mp4".format(d,camera_id,sequence_idx)
    vp_file    = "{}/DATA/vp/{}_axes.csv".format(d,camera_id)
    tf_file    = "{}/DATA/tform/{}_im_lmcs_transform_points.csv".format(d,camera_id)
    
    ann = Annotator_3D(video,label_file,tf_file,vp_file, load_corrected = True)
    ann.run()