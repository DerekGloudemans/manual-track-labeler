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
detector_path = os.path.join(os.getcwd(),"models","py_ret_det_multigpu")
sys.path.insert(0,detector_path)

MOT_util_path = os.path.join(os.getcwd(),"util_MOT")
sys.path.insert(0,MOT_util_path)

eval_path = os.path.join(os.getcwd(),"util_eval","py_motmetrics")
sys.path.insert(0,eval_path)


# filter and CNNs
from util_track.kf import Torch_KF
from models.py_ret_det_multigpu.retinanet.model import resnet50 


from util_eval import mot_eval as mot
from tracker_fsld import Localization_Tracker
from mock_detector import Mock_Detector

from util_MOT.MOT_detection_dataset import class_dict

def get_track_dict(TRAIN, dataset = "MOT20"):
    # get list of all files in directory and corresponding path to track and labels
    if TRAIN and dataset == "MOT20":
        data_dir = "/home/worklab/Data/cv/MOT20/train"
    elif (not TRAIN) and dataset == "MOT20":
        data_dir = "/home/worklab/Data/cv/MOT20/test"
    elif TRAIN and dataset == "MOT17":
        data_dir = "/home/worklab/Data/cv/MOT17/train_unique" 
    elif TRAIN and dataset == "MOT15":
        data_dir = "/home/worklab/Data/cv/MOT15/train" 
    elif (not TRAIN) and dataset == "MOT15":
        data_dir = "/home/worklab/Data/cv/MOT15/test"      
    elif TRAIN and dataset == "MOT16":
        data_dir = "/home/worklab/Data/cv/MOT16/train" 
    elif (not TRAIN) and dataset == "MOT16":
        data_dir = "/home/worklab/Data/cv/MOT16/test" 
    else:
        data_dir = "/home/worklab/Data/cv/MOT17/test_unique"
    
    track_list = [os.path.join(data_dir,item) for item in os.listdir(data_dir)]  
    track_dict = {}
    
    for item in track_list:
        if TRAIN:
            track_dict[item] = {
             "frames": os.path.join(item,"img1"),
             "labels":os.path.join(item,"gt/gt.txt")
                }
        else:
            track_dict[item] = {
             "frames": os.path.join(item,"img1")
                }
    
    return track_dict

def parse_gt_labels(label_path):
    class_dict = {
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
    
    ignored_regions = {}
    gts = {}
    
    with open(label_path,"r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            try:
                frame,obj_idx,left,top,width,height,conf,cls,visibility = row
                conf = float(conf)
                cls = torch.tensor(int(cls)).int()
                visibility = float(visibility)
                left = int(left)
                top = int(top)
                width = int(width)
                height = int(height)
                frame = int(frame)
                obj_idx = int(obj_idx)
                
            except: # MOT15
                frame,obj_idx,left,top,width,height,conf,_,_,_ = row
                cls = torch.tensor(1).int()
                frame = int(frame)
                obj_idx = int(obj_idx)
                left = float(left)
                top = float(top)
                width = float(width)
                height = float(height)
            
            bbox = torch.tensor([left,top,left+width,top+height]).int()
            
            if frame not in gts.keys():
                gts[frame] = []
                ignored_regions[frame] = []
            
            if cls in [2,7,8,12] or conf == 0:
                ignored_regions[frame].append(bbox)
        
            else:
                datum = {
                    "bbox": bbox,
                    "class_num": cls,
                    "id": obj_idx
                    }
                gts[frame].append(datum)
    
    all_keys = list(gts.keys())
    all_keys.sort()
    
    gts_list = []
    ignored_list = []
    for key in all_keys:    
        gts_list.append(gts[key])
        ignored_list.append(ignored_regions[key])
    
    return gts_list,ignored_list
            

if __name__ == "__main__":
      
     #add argparse block here so we can optinally run from command line
     #add argparse block here so we can optinally run from command line
     try:
        parser = argparse.ArgumentParser()
        parser.add_argument("det_steps",help = "list of values separated by commas")
        parser.add_argument("-confs", help = "list separated by commas",default = "0.95")
        parser.add_argument("-gpu",help = "gpu idx from 0-3", type = int,default = 0)
        parser.add_argument("-year",help = "15, 16, 17 or 20", type = int,default = 20)
        parser.add_argument("-mode",help = "iou or linear", type = str,default = "iou")
        parser.add_argument("-ber",help = "float >= 1.0", type = float,default = 1.1)
        parser.add_argument("-skip_steps",help = "list separated by commas",type = str, default = "1")
        parser.add_argument("-trims",help = "list separated by commas",type = str, default = "0")
        parser.add_argument("-fslds",help = "list separated by commas",type = str, default = "1")
        parser.add_argument("--test",action = "store_true")
        parser.add_argument("--show",action = "store_true")
        parser.add_argument("--public",action = "store_true")
        parser.add_argument("-save",help = "relative directory to save in", type = str,default = "results/temp")


        args = parser.parse_args()
        
        det_steps = args.det_steps.split(",")
        det_steps = [int(item) for item in det_steps]
        confs = args.confs.split(",")
        confs = [float(item) for item in confs]
        skip_steps = [int(item) for item in args.skip_steps.split(",")]
        
        

        TRAIN = not args.test
        GPU_ID = args.gpu
        dataset = "MOT{}".format(args.year)
        mode = args.mode
        ber = args.ber
        SHOW = args.show
        PUBLIC_DETECTIONS = args.public
        out_dir = args.save
        
        trims = args.trims.split(",")
        trims = [int(item) for item in trims]
        fslds = args.fslds.split(",")
        fslds = [int(item) for item in fslds]
        
     except:
         det_steps = [2]
         TRAIN = True
         skip_steps = [1]
         confs = [0.95]
         GPU_ID = 2
         dataset = "MOT20"
         mode = "linear"
         SHOW = False
         PUBLIC_DETECTIONS = False
         out_dir = "results/temp"
         fsld = 2 if mode == "linear" else 4 
         trim = 0
         
     tracker_name = "SORT" if mode == "linear" else "KIOU"
     print("\nStarting tracking run with the following settings:")
     print("Dataset:      {}".format(dataset))
     print("Training set: {}".format(TRAIN))
     print("Det steps:    {}".format(det_steps))
     print("Skip steps:   {}".format(skip_steps))
     print("Confs:        {}".format(confs))
     print("GPU ID:       {}".format(GPU_ID))
     print("FSLDs:         {}".format(fslds))
     print("Trims:         {}".format(trims))
     print("Tracker:      {}".format(tracker_name))
     print("Output directory: {}".format(out_dir))
     for det_step in det_steps:
         for det_conf_cutoff in confs:
             d = det_step             
             for skip_step in skip_steps:  
              for fsld in fslds:
               for trim in trims:
                det_step = d * skip_step
                
                subset = "train" if TRAIN else "test"
                
                
                #trim = 0
                # parameters
                #det_conf_cutoff = 0.95#0.45
                p_red = 30
                overlap = 0.5 # for evaluation, overlap with ignored areas
                iou_cutoff = 0.5 #0.9   # tracklets overlapping by this amount will be pruned
                #det_step = 4           # frames between full detection steps
                #skip_step = 1           # frames between update steps (either detection or localization)
                ber = 1.1 if dataset == "MOT20" else 1.6  # amount by which to expand a priori tracklet to crop object
                init_frames = 1         # number of detection steps in a row to perform
                matching_cutoff =  0.4 if mode == "iou" else 100 #for MOT20 detections farther from tracklets than this will not be matched 
                LOCALIZE = True         # if False, will skip frames between detection steps
                  # try 4 and 5 for MOT 20
                OUTVID = os.path.join(os.getcwd(),"demo","example_outputs")
                #mode = "iou"
                
                #
                loc_cp = "./config/localizer.pt"
                det_cp = "./config/detector.pt"

                class_dict = {
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
                
                # get filter
                filter_state_path = "./config/filter_params_tuned.cpkl"

                with open(filter_state_path ,"rb") as f:
                         kf_params = pickle.load(f)
                         #kf_params["R"] /= 100
                         kf_params["Q"] /= p_red
                         
                # get localizer
                localizer = resnet50(num_classes=14,device_id = GPU_ID)
                cp = torch.load(loc_cp)
                localizer.load_state_dict(cp) 
                
                # no localization update!
                if not LOCALIZE:
                    localizer = None
                
                # get detector
                detector = resnet50(num_classes=14,device_id = GPU_ID)
                try:
                    detector.load_state_dict(torch.load(det_cp))
                except:    
                    temp = torch.load(det_cp)["model_state_dict"]
                    new = {}
                    for key in temp:
                        new_key = key.split("module.")[-1]
                        new[new_key] = temp[key]
                    detector.load_state_dict(new)
                
                if PUBLIC_DETECTIONS:
                    detector = Mock_Detector("/home/worklab/Data/cv/MOT20/train")
                
                # get track_dict
                track_dict = get_track_dict(TRAIN,dataset = dataset)         
                tracks = [key for key in track_dict]
                tracks.sort()  
                
                
                # for each track and for specified det_step, track and evaluate
                running_metrics = {}
                for id in tracks:
                    
                    track_dir = track_dict[id]["frames"]
                    track_name = id.split("/")[-1].split(".")[0]
                    save_file = os.path.join(out_dir,"results_{}_{}_{}_{}_{}.cpkl".format(track_name,det_step,fsld,trim,int(det_conf_cutoff*10)))
                    
                    if PUBLIC_DETECTIONS:
                        detector.set_track(track_name)
                    
                    if os.path.exists(save_file):
                        continue

                    with open(save_file,"wb") as f:
                                        pickle.dump(0,f) # empty file to prevent other workers from grabbing this file as well
                    
                    
                    
                    # track it!
                    tracker = Localization_Tracker(track_dir,
                                                   detector,
                                                   localizer,
                                                   kf_params,
                                                   class_dict,
                                                   det_step = det_step,
                                                   skip_step = skip_step,
                                                   init_frames = init_frames,
                                                   fsld_max = fsld,
                                                   det_conf_cutoff = det_conf_cutoff,
                                                   matching_cutoff = matching_cutoff,
                                                   iou_cutoff = iou_cutoff,
                                                   ber = ber,
                                                   PLOT = SHOW,
                                                   downsample = 1,
                                                   distance_mode = mode,
                                                   cs = 112,
                                                   device_id = GPU_ID,
                                                   trim = trim)
                    
                    tracker.track()
                    preds, Hz, time_metrics = tracker.get_results()

                    if TRAIN:
                        # get ground truth labels
                        gts, ignored_regions = parse_gt_labels(track_dict[id]["labels"])
                        
                
                        # match and evaluate
                        metrics,acc = mot.evaluate_mot(preds,gts,ignored_regions,threshold = 0.5,ignore_threshold = overlap)
                        metrics = metrics.to_dict()
                        metrics["framerate"] = {0:Hz}
                        print("\rsequence {} ---> MOTA: {}  Framerate: {}".format(track_name,np.round(metrics["mota"][0],4),np.round(metrics["framerate"][0],2)))
                    else:
                        metrics = None
                    
                    with open(save_file,"wb") as f:
                        pickle.dump((preds,metrics,time_metrics,Hz),f)
                    
                    if TRAIN:
                        # add results to aggregate results
                        try:
                            for key in metrics:
                                running_metrics[key] += metrics[key][0]
                        except:
                            for key in metrics:
                                running_metrics[key] = metrics[key][0]
                                
                if TRAIN:            
                    # average results  
                    print("\nAverage Metrics for detector with {} tracks with det_step {}, fsld {}, trim {}".format(len(tracks),det_step,fsld,trim))
                    for key in running_metrics:
                        running_metrics[key] /= len(tracks)
                        print("   {}: {}".format(key,running_metrics[key]))
                    print("")