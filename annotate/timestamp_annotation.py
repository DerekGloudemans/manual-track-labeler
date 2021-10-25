#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:16:25 2021

@author: worklab
"""
import os
import re
import cv2
import ast
import numpy as np
import _pickle as pickle


def get_precomputed_checksums(abs_path=None):
    path = './resources/timestamp_pixel_checksum_6.pkl'
    if abs_path is not None:
        path = abs_path
        
    with open(path, 'rb') as pf:
        dig_cs6 = pickle.load(pf)
        
    return dig_cs6


def get_timestamp_geometry(abs_path=None):
    path = './resources/timestamp_geometry_4K.pkl'
    if abs_path is not None:
        path = abs_path
        
    with open(path, 'rb') as pf:
        g = pickle.load(pf)
    return g


def get_timestamp_pixel_limits():
    """
    Provides x/y coordinates (only) for timestamp pixel extraction. Note that return order is y1, y2, x1, x2. Timestamp
        can be extracted from full frame like so: `timestamp_pixels = frame_pixels[y1:y2, x1:x2, :]`
    :return: y-start (y1), y-stop (y2), x-start (x1), x-stop (x2)
    """
    geom = get_timestamp_geometry()
    y0 = geom['y0']
    x0 = geom['x0']
    h = geom['h']
    n = geom['n']
    w = geom['w']
    return y0, y0+h, x0, x0+(n*w)


def parse_frame_timestamp(timestamp_geometry, precomputed_checksums, frame_pixels=None, timestamp_pixels=None,thresh = 127,Examine = False):
    """
    Use pixel checksum method to parse timestamp from video frame. First extracts timestamp area from frame
        array. Then converts to gray-scale, then converts to binary (black/white) mask. Each digit
        (monospaced) is then compared against the pre-computed pixel checksum values for an exact match.
    :param timestamp_geometry: dictionary of parameters used for determining area of each digit in checksum
        (load using utilities.get_timestamp_geometry)
    :param precomputed_checksums: dictionary of checksum:digit pairs (load using utilities.get_precomputed_checksums())
    :param frame_pixels: numpy array of full (4K) color video frame; dimensions should be 2160x3840x3
    :param timestamp_pixels: numpy array of timestamp area, defined by `get_timestamp_pixel_limits()`
    :return: timestamp (None if checksum error), pixels from error digit (if no exact checksum match)
    """
    # extract geometry values from timestamp_geometry dictionary
    g = timestamp_geometry
    w = g['w']
    h = g['h']
    x0 = g['x0']
    y0 = g['y0']
    n = g['n']
    h13 = g['h13']
    h23 = g['h23']
    h12 = g['h12']
    w12 = g['w12']
        
    dig_cs6 = precomputed_checksums
    
    if frame_pixels is not None:
        # extract the timestamp in the x/y directions
        tsimg = frame_pixels[y0:(y0+h), x0:(x0+(n*w)), :]
    elif timestamp_pixels is not None:
        tsimg = timestamp_pixels
    else:
        raise ValueError("One of `frame_pixels` or `timestamp_pixels` must be specified.")
    # convert color to gray-scale
    tsgray = cv2.cvtColor(tsimg, cv2.COLOR_BGR2GRAY)
    # convert to black/white binary mask using fixed threshold (1/2 intensity)
    # observed gray values on the edges of some digits in some frames were well below this threshold
    ret, tsmask = cv2.threshold(tsgray, thresh, 255, cv2.THRESH_BINARY)

    # parse each of the `n` digits
    ts_dig = []
    for j in range(n):
        # disregard the decimal point in the UNIX time (always reported in .00 precision)
        if j == 10:
            ts_dig.append('.')
            continue

        # extract the digit for this index, was already cropped x0:x0+n*w, y0:y0+h
        pixels = tsgray[:, j*w:(j+1)*w]
        
        
        
        # compute the 6-area checksum and convert it to an array
        cs = [[(pixels[:h13, :w12].sum() / 255), (pixels[:h13, w12:].sum() / 255)],
              [(pixels[h13:h23, :w12].sum() / 255), (pixels[h13:h23, w12:].sum() / 255)],
              [(pixels[h23:, :w12].sum() / 255), (pixels[h23:, w12:].sum() / 255)]
              ]
        cs = np.array(cs)
        
        if Examine: # for matching timestamp pixels to checksums
            #print(cs ,"\n")
            cv2.imshow("frame",tsgray[:, j*w:(j+1)*w])
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if key == ord("0"):
                temp[0].append(cs)
            elif key == ord("1"):
                temp[1].append(cs)  
            elif key == ord("2"):
                temp[2].append(cs)
            elif key == ord("3"):
                temp[3].append(cs)
            elif key == ord("4"):
                temp[4].append(cs)
            elif key == ord("5"):
                temp[5].append(cs) 
            elif key == ord("6"):
                temp[6].append(cs)
            elif key == ord("7"):
                temp[7].append(cs) 
            elif key == ord("8"):
                temp[8].append(cs)
            elif key == ord("9"):
                temp[9].append(cs)
        
        # compute the absolute difference between this digit and each candidate
        cs_diff = [(dig, abs(cs - cs_ref).sum()) for dig, cs_ref in dig_cs6.items()]
        pred_dig, pred_err = min(cs_diff, key=lambda x: x[1])
        #looking for a perfect checksum match; testing showed this was reliable
        if pred_err > 1:
            #if no exact match, return no timestamp and the pixel values that resulted in the error
            return None, pixels
        else:
            ts_dig.append(pred_dig)
    # convert the list of strings into a number and return successful timestamp
    return ast.literal_eval(''.join(map(str, ts_dig))), None


temp = dict([(i,[]) for i in range(10)])

test_vid = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments/p1c2_0.mp4"

cap = cv2.VideoCapture(test_vid)

ret,frame = cap.read()

with open ("/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_geometry_4K.pkl", "rb") as f:
    geom = pickle.load(f)
    
    
with open ("/home/worklab/Documents/derek/i24-dataset-gen/annotate/grayscale_checksums_1080p.pkl", "rb") as f:
    checksums = pickle.load(f)
    
    
#slice timestamp from frame
    
        
    x0 = int(geom["x0"]/2.0)
    y0 = int(geom["y0"]/2.0)
    x1 = int(geom["x0"]/2.0 + geom["n"] * geom["w"]/2.0)
    y1 = int(geom["y0"]/2.0 + geom["h"]/2.0)
    
    geom_half = dict([(key,int(geom[key]/2.0)) for key in geom])
    geom_half["n"] = int(geom["n"])
    
    out = (0,0)
    while True:
        
        timestamp = frame[y0:y1,x0:x1,:]
        timestamp_4k = cv2.resize(timestamp,(timestamp.shape[1]*2,timestamp.shape[0]*2))
        
        timestamp_big = cv2.resize(timestamp,(timestamp.shape[1]*10,timestamp.shape[0]*10))
        cv2.imshow("ts",timestamp)
        key = cv2.waitKey(1)
        Examine = False
        if key == ord("e"):
            Examine = True
        elif key == ord("q"):
            break
        
        
        prev_out = out
        out = parse_frame_timestamp(geom_half,checksums,timestamp_pixels = timestamp,thresh = 80,Examine = Examine)
        if np.abs(out[0] - prev_out[0]) > 0.04:
            print("Timestamp: {} ERROR".format(out))
        else:
            print("Timestamp: {}".format(out))
        
        cv2.destroyAllWindows()
        ret,frame = cap.read()
        
    
    cap.release()
    
    for key in temp:
        temp[key] = sum(temp[key]) / len(temp[key])
    
    # with open("grayscale_checksums_1080p.pkl","wb") as f:
    #     pickle.dump(temp,f)
        