import cv2
import cv2 as cv
import time
import os
import numpy as np 
import csv
import string


directory = "/home/worklab/Data/cv/video/5_min_18_cam_October_2020/ingest_session_00005/recording"
directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/trimmed"

ds = 2

all_ims = []
used_cameras = []
dir_list = os.listdir(directory)
dir_list.sort()
for item in dir_list:
    camera_name = item.split("_")[0]
    homography_name = "{}_im_lmcs_transform_points.csv".format(camera_name)
    if  homography_name in os.listdir("tform") and camera_name not in used_cameras:
        used_cameras.append(camera_name)
        # get frame
        video = os.path.join(directory,item)
        cap = cv2.VideoCapture(video)
        ret,frame = cap.read()
        if ds != 1:
            frame = cv2.resize(frame,(frame.shape[1]//ds,frame.shape[0]//ds))
                
        # get tform
        keep = []
        with open(os.path.join("tform",homography_name),"r") as f:
            read = csv.reader(f)
            SKIP = True
            for row in read:
                
                if SKIP and "im space" not in row[0]:
                    continue
                
                
                if not SKIP:
                    keep.append(row)
                        
                if "im space"  in row[0]:
                    SKIP = False
                    
        H = np.stack([[float(item) for item in row] for row in keep])
        
        warped_im = cv2.warpPerspective(frame, H,(2000,60))
        # cv2.imshow("frame",warped_im)
        # cv2.waitKey(2)
        # cv2.destroyAllWindows()
        all_ims.append(warped_im)

base_im = all_ims[0].copy()
for i in range(1,len(all_ims)):
    base_im = np.array(base_im)
    # get a mask of all zero pixel in base image
    # mask =  1- np.clip(base_im.copy(),0,1)
    # comb_im = cv2.multiply(np.array(all_ims[i].copy()),mask)
    
    # cv2.imshow("combo",comb_im.astype(np.int8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    #base_im = base_im + comb_im
    
    combined = np.where(base_im > 0,base_im,all_ims[i].copy())
    base_im = combined.copy()
    # cv2.imshow("combined",base_im)
    # cv2.waitKey(0)
    
    #base_im = cv2.add(base_im,all_ims[i].copy().astype(np.int8),mask = mask)
    
combo = base_im.copy()
combo = cv2.resize(combo, (combo.shape[1]*2,combo.shape[0]*2))
combo = cv2.flip(combo,0)
cv2.imshow("frame",combo)
cv2.waitKey(0)
cv2.imwrite("combined.png",combo)
cv2.destroyAllWindows()
