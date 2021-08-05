import csv
import numpy as np
import multiprocessing as mp
import cv2
import queue
import time
import os


directory = "/home/worklab/Data/dataset_alpha/trial1"

for file in os.listdir(directory):
    camera = file.split("_")[0]
    file = os.path.join(directory,file)
    HEADERS = True
    
    out_rows = []
    with open(file,"r") as f:
        read = csv.reader(f)
        
        for row in read:
            length = len(row)
            
            if length != 44:
                
                # check if it's the header line
                if length > 0:
                    
                    if row[0] == "Frame #":
                        try:
                            del row[12]
                            del row[11]
                        except:
                            pass
                        row = row + ["frx","fry","flx","fly","brx","bry","blx","bly","direction","camera","acceleration","speed","x","y","theta","width","length"]
                        HEADERS = False
                    else:
                        try:
                            del row[12]
                            del row[11]
                        except:
                            pass
                        pad = ["" for i in range(44-length)]
                        row = row + pad
                        
                        if not HEADERS:
                            row[36] = camera
                    
                    
            out_rows.append(row)
                
    
    with open(file,"w") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
    
    print("Reformatted file {}".format(file))