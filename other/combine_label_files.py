import csv
import numpy as np
import multiprocessing as mp
import cv2
import queue
import time
import os
import time


    
    
def combine_csvs(file1,file2,cutoff_frame = 0):
    
    short_name = file1.split("/")[-1]
    HEADERS = True
    
    # parse first file
    rows = []
    with open(file1,"r") as f:
        read = csv.reader(f)
        
        for row in read:
            rows.append(row)
                
    out_rows = []
    HEADERS = True
    for row_idx in range(len(rows)):
        row = rows[row_idx]
        
        # pass header lines through as-is
        if HEADERS:
            out_rows.append(row)
            if len(row) > 0 and row[0] == "Frame #":
                HEADERS = False
        
        
        else:
            
            if len(row) == 0:
                continue
            
            if int(row[0]) < cutoff_frame:
                out_rows.append(row)
            else:
                break
        
        
    # parse second file
    rows = []
    with open(file2,"r") as f:
        read = csv.reader(f)
        
        for row in read:
            rows.append(row)
                
    HEADERS = True
    for row_idx in range(len(rows)):
        row = rows[row_idx]
        
        # pass header lines through as-is
        if HEADERS:
            if len(row) > 0 and row[0] == "Frame #":
                HEADERS = False
        else:
            if int(row[0]) >= cutoff_frame:
                out_rows.append(row)

        
    
    
    outfile = "/" + os.path.join(*file1.split("/")[:-2]) + "/combined/" + short_name
    with open(outfile,"w") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
    
    

if __name__ == "__main__":
    
    cutoff_frame = 400
    manual =    "/home/worklab/Desktop/data_port/FOR ANNOTATORS/rectified_p1c6_0_track_outputs_3D.csv"
    rectified = "/home/worklab/Desktop/data_port/Rectified_3D/rectified_p1c6_0_track_outputs_3D.csv"
    combine_csvs(manual,rectified,cutoff_frame)