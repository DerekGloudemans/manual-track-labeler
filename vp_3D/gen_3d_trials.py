import cv2
import cv2 as cv
import time
from os import mkdir
import os
from PIL import Image 
import numpy as np 
import csv
import sys
import torch
import torchvision.transforms.functional as F
from torchvision.ops import nms
import time
import math
from scipy.spatial import ConvexHull

detector_path = os.path.join(os.getcwd(),"py_ret_det_multigpu")
sys.path.insert(0,detector_path)
from py_ret_det_multigpu.retinanet.model import resnet50 

from axis_labeler import Axis_Labeler
from utils import * # ...oooooooops...


def detect_3D(video_sequence,avg_frame = None, vps = None,ds = 1):
    
    # init detector
    GPU_ID = 3
    detector = resnet50(num_classes=13,device_id = GPU_ID)
    cp = torch.load("./config/detector.pt")
    detector.load_state_dict(cp) 
    detector = detector.to(GPU_ID)
    detector.eval()
    detector.training = False
    
    # open up a videocapture object
    cap = cv2.VideoCapture(video_sequence)
    # verify file is opened
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_sequence)
    
    # get first frame
    start = time.time()
    frame_num = 0
    ret, frame = cap.read()
    
    # downsample first frame
    if ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//ds,frame.shape[0]//ds))
    
    # resize average frame
    if avg_frame is None:
        avg_frame = np.zeros(frame.shape).astype(np.uint8)
    avg_frame = cv2.resize(avg_frame,(frame.shape[1],frame.shape[0]))
        
    
    ### MAIN LOOP, one iteration per frame
    while ret: 
        frame_num += 1
        
        out_name = "output/{}.png".format(str(frame_num).zfill(4))
        if frame_num > -1:
            # get detections
            im = np.array(frame)[:,:,[2,1,0]].copy()
            im = F.to_tensor(im)
            im = F.normalize(im,mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            
            device = torch.device("cuda:{}".format(GPU_ID))
            im = im.to(device).unsqueeze(0)
            scores,_,boxes = detector(im)
            
            # nms across all classes
            idxs = nms(boxes,scores,0.3)
            boxes = boxes[idxs]
            boxes = boxes.cpu().data.numpy()
    
            
            # get features
            # edges = cv2.Canny(frame,250,200)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            # thresh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            diff = np.clip(np.abs(frame.astype(int) - avg_frame.astype(int)),0,255)
    
            # kernel to remove small noise
            diff = cv2.blur(diff,(5,5)).astype(np.uint8)
            
           
            # threshold
            _,diff = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)
            diff =  diff.astype(np.uint8) # + cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
            
            # cv2.imshow("diff",diff)
            # cv2.waitKey(0)
            
            if vps is not None and len(boxes) > 0 and frame_num > 20:
                #try:
                    boxes_3D = fit_3D_boxes(diff,boxes, vps[0], vps[1], vps[2], show = False,verbose = False, granularity = 1e-02,e_init = 3e-02)
                # except Exception as E:
                #     print(E)
                #     boxes_3D = []
            else:
                boxes_3D = []
            if True:
                #plot 2D bboxes
                for box in boxes:
                    color = (255,0,255) #colors[int(obj.cls)]
                    c1 =  (int(box[0]),int(box[1]))
                    c2 =  (int(box[2]),int(box[3]))
                    frame = cv2.rectangle(frame,c1,c2,color,1)
        
            
                # plot 3D bboxes
                if "Error" not in boxes_3D:
                    for box in boxes_3D:
                        #frame = plot_3D(frame,box,vp1,vp2,vp3,threshold = 200)
                        frame = plot_3D_ordered(frame,box)

            
    
        cv2.imshow("3D Estimated Bboxes",frame)
        cv2.waitKey(1)
        #cv2.imwrite(out_name,frame)
            
        print("\r On frame: {}, FPS: {:5.2f}".format(frame_num, frame_num / (time.time() - start)),end = '\r', flush = True)
        torch.cuda.empty_cache()
        
        # get next frame
        ret, frame = cap.read()
        
        if ret and ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//ds,frame.shape[0]//ds))
        
        if frame_num > 1000: # early video cutoff
            cap.release()
            break
       
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_sequence = "/home/worklab/Desktop/test_vid.mp4"
    test_sequence = "/home/worklab/Data/cv/video/5_min_18_cam_October_2020/ingest_session_00005/recording/record_p1c5_00001.mp4"
    test_sequence = "/home/worklab/Data/cv/video/ground_truth_video_06162021/trimmed/p1c1_trimmed.mp4"

    downsample = 2
    
    # get average frame
    try:
        name =  "config/" + test_sequence.split("/")[-1].split(".mp4")[0].split("_")[1] + "_avg.png"
        avg = cv2.imread(name)
        if avg is None:
            raise FileNotFoundError
    except:
        avg = get_avg_frame(test_sequence,ds = downsample).astype(np.uint8)
        name = "config/" + test_sequence.split("/")[-1].split(".mp4")[0].split("_")[1] + "_avg.png"
        cv2.imwrite(name,avg)
   
    
    
    
    # get axes annotations
    try:
        name = "config/" + test_sequence.split("/")[-1].split(".mp4")[0].split("_")[1] + "_axes.csv"
        labels = []
        with open(name,"r") as f:
            read = csv.reader(f)
            for row in read:
                if len(row) == 5:
                    row = [int(float(item)) for item in row]
                elif len(row) > 5:
                    row = [int(float(item)) for item in row[:5]] + [float(item) for item in row[5:]]
                labels.append(np.array(row))
                        
    except FileNotFoundError:
        labeler = Axis_Labeler(test_sequence,ds = downsample)
        labeler.run()
        labels = labeler.axes
    
    
    # get vanishing points
    if True:    
        lines1 = []
        lines2 = []
        lines3 = []
        for item in labels:
            if item[4] == 0:
                lines1.append(item)
            elif item[4] == 1:
                lines2.append(item)
            elif item[4] == 2:
                lines3.append(item)
        
        # get all axis labels for a particular axis orientation
        vp1 = find_vanishing_point(lines1)
        vp2 = find_vanishing_point(lines2)
        vp3 = find_vanishing_point(lines3)
        
        plot_vp(test_sequence,vp1 = vp1,vp2 = vp2,vp3 = vp3, ds  = downsample)
        
        
        
    # detect and convert
    detect_3D(test_sequence,avg_frame = avg, vps = [vp1,vp2,vp3],ds = downsample)
      
    cv2.destroyAllWindows()
    
    
