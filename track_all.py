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
from tracker_fsld import Localization_Tracker           

if __name__ == "__main__":
      
     #add argparse block here so we can optinally run from command line
     #add argparse block here so we can optinally run from command line
     try:
        parser = argparse.ArgumentParser()
        parser.add_argument("directory",help = "path to video directory")
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
         GPU_ID = 0
         SHOW = True
         out_dir = "output"

         
     print("\nStarting tracking run with the following settings:")
     print("GPU ID:       {}".format(GPU_ID))
     print("Output directory: {}".format(out_dir))
     OUTVID = None
     
                
                
     if True:
        loc_cp             = "./_config/localizer_retrain_112.pt"
        det_cp             = "./_config/i24_detector_old.pt"
        filter_state_path  = "./_config/filter_params_tuned.cpkl"
        config             = "./_config/lbt_params.config"
        
        class_dict = {
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
        
        # get filter    
        with open(filter_state_path ,"rb") as f:
                 kf_params = pickle.load(f)
                 
        # get localizer
        localizer = resnet50(num_classes=5,device_id = GPU_ID)
        cp = torch.load(loc_cp)
        localizer.load_state_dict(cp) 

        # get detector
        detector = resnet50(num_classes=9,device_id = GPU_ID)
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
        
        for sequence in tracks:           
            
            track_name = sequence.split("/")[-1].split(".")[0]
            save_file = os.path.join(out_dir,"results_{}.cpkl".format(track_name))
            
            
            if os.path.exists(save_file):
                continue
    
            # with open(save_file,"wb") as f:
            #                     pickle.dump(0,f) # empty file to prevent other workers from grabbing this file as well
            
            
            
            # track it!
            tracker = Localization_Tracker(sequence,
                                           detector,
                                           localizer,
                                           kf_params,
                                           class_dict,
                                           config,
                                           PLOT = SHOW,
                                           device_id = GPU_ID,
                                           OUT = OUTVID)
            
            tracker.track()
            preds, Hz, time_metrics = tracker.get_results()
            print("\rsequence {} --->  Framerate: {}".format(track_name,np.round(np.round(Hz,2))))
            
            with open(save_file,"wb") as f:
                pickle.dump((preds,time_metrics,Hz),f)
            
