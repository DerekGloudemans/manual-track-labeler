import csv
import numpy as np
import multiprocessing as mp
import cv2
import queue
import time
import os
import time


    
    
def interpolate_csv(file):
    interp_f = 20
    
    
    camera = file.split("/")[-1].split("_")[0]
    HEADERS = True
    
    rows = []
    with open(file,"r") as f:
        read = csv.reader(f)
        
        for row in read:
            rows.append(row)
                
    count = 0
    start_time = time.time()
    frame_idx = 0
    
    obj_cache = {} # will cache prev_box_frame, prev_box, next_box_frame,next_box
    
    out_rows = []
    HEADERS = True
    for row_idx in range(len(rows)):
        row = rows[row_idx]
        
        # pass header lines through as-is
        if HEADERS:
            if len(row) > 0 and row[0] == "Frame #":
                HEADERS = False
        else:
            if row[11] == '': # there is no 3D box for this frame and obj idx
                
                # search forwards and backwards up to interp_f frames for a box corresponding to this obj idx
                obj_idx = int(row[2])
                frame_idx = int(row[0])
                
                back_box = None
                if obj_idx in obj_cache.keys():
                    cache = obj_cache[obj_idx]
                    
                    if cache[0] >= frame_idx - interp_f:
                        back_box = cache[1]
                        back_frame_idx = cache[0]
                
                # else:
                #     # search backwards for this obj_idx
                #     back_row_idx = row_idx -1
                #     back_frame_idx = frame_idx
                #     back_box = None
                #     while back_row_idx > 0 and back_frame_idx >= frame_idx - interp_f:
                #         back_row = rows[back_row_idx]
                        
                #         back_obj_idx = int(back_row[2])
                #         back_frame_idx = int(back_row[0])
                        
                #         if back_obj_idx == obj_idx and back_row[11] != '':
                #             back_box = np.array(back_row[11:27]).astype(float)
                #             break
                #         else:
                #             back_row_idx -= 1
                
                fore_box = None
                if back_box is not None:
                    if obj_idx in obj_cache.keys():
                        if cache[2] > frame_idx:
                            fore_frame_idx = cache[2]
                            fore_box = cache[3]
                    
                    if fore_box is None:
                        # search forwards for this obj_idx
                        fore_row_idx = row_idx +1
                        fore_frame_idx = frame_idx
                        fore_box = None
                        while fore_row_idx < len(rows) and fore_frame_idx <= frame_idx + interp_f:
                            fore_row = rows[fore_row_idx]
                            
                            fore_obj_idx = int(fore_row[2])
                            fore_frame_idx = int(fore_row[0])
                            
                            if fore_obj_idx == obj_idx and fore_row[11] != '':
                                fore_box = np.array(fore_row[11:27]).astype(float)
                                break
                            else:
                                fore_row_idx += 1
                        
                if back_box is not None and fore_box is not None:
                    interp_box = fore_box * (frame_idx-back_frame_idx)/(fore_frame_idx-back_frame_idx) + back_box * (fore_frame_idx-frame_idx)/(fore_frame_idx-back_frame_idx)
                    
                    obj_cache[obj_idx] = [frame_idx,interp_box,fore_frame_idx,fore_box]
                    
                    row[11:27] = list(interp_box)
                    row[10] = "interp 3d"
                    count += 1
            else:
                obj_idx = int(row[2])
                frame_idx = int(row[0])
                if obj_idx in obj_cache.keys():
                    obj_cache[obj_idx][0:2]  = [frame_idx,np.array(row[11:27]).astype(float)]
                else:
                    obj_cache[obj_idx] = [frame_idx,np.array(row[11:27]).astype(float),-1,None]
    
        out_rows.append(row)
        print("\rOn frame {}".format(frame_idx), end = '\r', flush = True)

    
    
    with open(file,"w") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
    
    elapsed = int(time.time() - start_time)
    print("Interpolated {} boxes in {}s for file {}".format(count,elapsed,file))
    

if __name__ == "__main__":
    
    directory = "/home/worklab/Data/dataset_alpha/trial2"
    interp_f = 5
    for file in os.listdir(directory):
        file = os.path.join(directory,file)
        interpolate_csv(file)
        
    # file_list = [os.path.join(directory,file) for file in os.listdir(directory)]        
    # pool = mp.Pool(processes = 30)
    # pool.map(interpolate_csv,file_list)