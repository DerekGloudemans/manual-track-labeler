#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:59:25 2021

@author: worklab
"""


import torch
import time

time.sleep(5)

n_samples = 200

times = {
    '0-1':[],
    '0-2':[],
    '0-3':[],
    '1-2':[],
    '1-3':[],
    '2-3':[]
    }

for i in range(n_samples):
    
    for a in [1,2]:
        for b in [1,2,3]:
            
            if a < b:
                transfer = "{}-{}".format(a,b)
                
                device_a = "cuda:{}".format(a)
                device_b = "cuda:{}".format(b)
                
                im = torch.rand([6,3,3840,2160],dtype = float,device = device_a)
                torch.cuda.synchronize()
                
                start = time.time()
                
                im_copy = im.to(device_b)
                im_copy += 0.01
                torch.cuda.synchronize()
                end = time.time()
                
                times[transfer].append(end-start)
                time.sleep(0.01)
                
                del im 
                del im_copy
    
    if i % 50 == 0:
        print("finished iteration {}".format(i))
                

for key in times:
    a = key[0]
    b = key[2]
    time = times[key]
    ms = sum(time)/n_samples * 1000
    print("Average transfer latency from device {} to {}: {}ms".format(a,b,ms))