import cv2
import cv2 as cv
import time
import os
import numpy as np 
import csv

class Ignore_Labeler():
    def __init__(self,sequence, ds = 1):
        self.ds = ds
        
        self.sequence = sequence
        
        self.cap = cv2.VideoCapture(sequence)
        ret,frame = self.cap.read()
        if self.ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//self.ds,frame.shape[0]//self.ds))
        self.frames = [frame]
        
        
        self.points = []
        
        #self.load_annotations()
        
        self.cur_image = self.frames[0]
        
        self.start_point = None # used to store click temporarily
        self.clicked = False
        self.new = None # used to store a new box to be plotted temporarily
        self.cont = True
        self.define_direction = False
        
        # classes

        self.plot()
        self.changed = False

        
    def plot(self):
        self.cur_image = self.frames[0].copy()
        if len(self.points) < 3:
            return
        
        pts = np.array(self.points)
        pts = pts[np.newaxis,:]
        self.cur_image = cv2.fillPoly(self.cur_image,pts,(0,0,0))

    def on_mouse(self,event, x, y, flags, params):
    
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         point = np.array([x,y]).astype(np.int32)
         self.points.append(point)
         self.changed = True
         
         if len(self.points) > 2:
             self.plot()
                
            
    def quit(self):
        cv2.destroyAllWindows()
        self.cont = False
        self.save()
        
    def save(self):
        if len(self.points) == 0:
            return
        sequence = self.sequence.split(".")[0].split("/")[-1].split("_")[0]
        name = "ignored_regions/" + sequence + "_ignored.csv"
        if os.path.exists(name): # don't overwrite
            overwrite = input("Overwrite existing file? (y/n)")
            if overwrite != "y":
                return
        
        outputs = []
        for item in self.points:
            output = list(item)
            outputs.append(output)
        
        with open(name,"w") as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(outputs)
        print("Saved ignored_region as file {}".format(name))
        
        name = "ignored_regions" + name.split(".csv")[0].split("_")[1] + ".png"
        cv2.imwrite(name,self.cur_image)
        
    def undo(self):
        self.clicked = False
        self.define_direction = False
        self.define_magnitude = False
        
        self.points = self.points[:-1]
        self.cur_image = self.frames[0].copy()
        self.plot()
        
        
    def run(self):  
        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
           
        while(self.cont): # one frame
                    
           cv2.imshow("window", self.cur_image)
           title = "Label 4 corners that define the ignored region polygon ({})".format(len(self.points))
           cv2.setWindowTitle("window",str(title))
           
           key = cv2.waitKey(1)
           if key == ord("q"):
                self.quit()

           elif key == ord("u"):
               self.undo()



if __name__ == "__main__":
    
    directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments/"
    all_sequences = [os.path.join(directory,sequence) for sequence in os.listdir(directory)]
    
    for sequence in all_sequences:
        # check if this camera has already been done
        cam_name = sequence.split("/")[-1].split("_")[0]
        
        ignored_file_name = "ignored_regions/{}_ignored.csv".format(cam_name)
        if os.path.exists(ignored_file_name):
            continue
        
        labeler = Ignore_Labeler(sequence)
        labeler.run()