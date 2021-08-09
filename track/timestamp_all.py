import argparse
import os,sys,inspect
import numpy as np
import random 
import time
import math
import csv
import _pickle as pickle
random.seed = 0

import cv2
from PIL import Image
import torch

import matplotlib.pyplot  as plt

# add relevant packages and directories to path
detector_path = os.path.join(os.getcwd(),"model","py_ret_det_multigpu")
sys.path.insert(0,detector_path)

# filter and CNNs
from model.py_ret_det_multigpu.retinanet.model import resnet50 
from timestamp_parser import Localization_Tracker           

if __name__ == "__main__":
      
     #add argparse block here so we can optinally run from command line
     #add argparse block here so we can optinally run from command line
     try:
        directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"


        parser = argparse.ArgumentParser()
        parser.add_argument("-directory",help = "path to video directory",default = directory)
        parser.add_argument("-gpu",help = "gpu idx from 0-3", type = int,default = 0)
        parser.add_argument("--show",action = "store_true")
        parser.add_argument("-save",help = "relative directory to save in", type = str,default = "output")


        args = parser.parse_args()
        directory = args.directory
        GPU_ID = args.gpu
        SHOW = args.show
        out_dir = args.save

       
        
     except:
         directory = "/home/worklab/Data/cv/video/5_min_18_cam_October_2020/ingest_session_00005/recording"
         directory = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k"
         GPU_ID = 0
         SHOW = True
         

     out_dir = "/home/worklab/Data/dataset_alpha/track_2d"    
     print("\nStarting tracking run with the following settings:")
     print("GPU ID:       {}".format(GPU_ID))
     print("Output directory: {}".format(out_dir))
     OUTVID = "vid"
     
                
                
     if True:
        loc_cp             = "./_config/localizer_april_112.pt"
        det_cp             = "./_config/detector_april.pt"
        filter_state_path  = "./_config/filter_params_tuned.cpkl"
        config             = "./_config/lbt_params.config"
        
        class_dict = { "sedan":0,
                    "midsize":1,
                    "van":2,
                    "pickup":3,
                    "semi":4,
                    "truck (other)":5,
                    "motorcycle":6,
                    "trailer":7,
                    0:"sedan",
                    1:"midsize",
                    2:"van",
                    3:"pickup",
                    4:"semi",
                    5:"truck (other)",
                    6:"motorcycle",
                    7:"trailer",
                    }
        
        # get filter    
        with open(filter_state_path ,"rb") as f:
                 kf_params = pickle.load(f)
                 
        # get localizer
        localizer = resnet50(num_classes=8,device_id = GPU_ID)
        cp = torch.load(loc_cp)
        localizer.load_state_dict(cp) 

        # get detector
        detector = resnet50(num_classes=8,device_id = GPU_ID)
        try:
            detector.load_state_dict(torch.load(det_cp))
        except:    
            temp = torch.load(det_cp)["model_state_dict"]
            new = {}
            for key in temp:
                new_key = key.split("module.")[-1]
                new[new_key] = temp[key]
            detector.load_state_dict(new)

        
        # for each track and for specified det_step, track and evaluate
        tracks = [os.path.join(directory,item) for item in os.listdir(directory)]
        tracks.sort()
        
        
        all_timestamps = {}
        for sequence in tracks:           
            
            track_name = sequence.split("/")[-1].split(".")[0]

            
            # track it!
            tracker = Localization_Tracker(sequence,
                                           detector,
                                           localizer,
                                           kf_params,
                                           class_dict,
                                           config,
                                           PLOT = SHOW,
                                           out_dir = out_dir,
                                           device_id = GPU_ID,
                                           OUT = OUTVID)
            
            sequence_timestamps = tracker.track()
            all_timestamps[track_name] = sequence_timestamps
            
            print("\nsequence {} done".format(track_name))
            
            # with open(save_file,"wb") as f:
            #     pickle.dump((preds,time_metrics,Hz),f)
            
            
            with open("saved_alpha_timestamps.cpkl","wb") as f:
                pickle.dump(all_timestamps,f)
            
