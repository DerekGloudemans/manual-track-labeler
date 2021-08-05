import csv
import numpy as np
import multiprocessing as mp
import cv2
import queue
import time


def plot_rectified_objects(
        sequence,
        csv_file,
        frame_rate = 10.0,
        show_2d = False,
        show_3d = False,
        show_LMCS = False,
        show_rectified = False,
        save = False
        ):
        
    
    class_colors = [
            (0,255,0),
            (255,0,0),
            (0,0,255),
            (255,255,0),
            (255,0,255),
            (0,255,255),
            (255,100,0),
            (255,50,0),
            (0,255,150),
            (0,255,100),
            (0,255,50)]
    
    classes = { "sedan":0,
                    "midsize":1,
                    "van":2,
                    "pickup":3,
                    "semi":4,
                    "truck (other)":5,
                    "truck": 5,
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
    
    # get the camera id - only these boxes will be parsed from the data file
    camera_id = sequence.split("/")[-1].split("_")[0]
    if camera_id[0] != "p":
        camera_id = sequence.split("/")[-1].split("_")[1]
        if camera_id[0] != "p":
            print("Check sequence naming, cannot find camera id")
            return
    relevant_camera = camera_id
    
    # load LMCS -> im space homography matrix
    transform_path = "./annotate/tform/{}_im_lmcs_transform_points.csv".format(relevant_camera)
    # get transform from RWS -> ImS
    keep = []
    with open(transform_path,"r") as f:
        read = csv.reader(f)
        FIRST = True
        for row in read:
            if FIRST:
                FIRST = False
                continue
                    
            if "im space"  in row[0]:
                break
            
            keep.append(row)
    pts = np.stack([[float(item) for item in row] for row in keep])
    im_pts = pts[:,:2]
    lmcs_pts = pts[:,2]
    H,_ = cv2.findHomography(lmcs_pts,im_pts)
    
    
    # store each item by frame_idx
    all_frame_data = {}
    
    # store estimated heights and classes per object idx
    obj_heights = {}
    obj_cls = {}
    

    # loop through CSV file    
    frame_labels = {}
    with open(csv_file,"r") as f:
        read = csv.reader(f)
        HEADERS = True
        for row in read:
            
            #get rid of first row
            if HEADERS:
                HEADERS = False
                continue
                
            camera = row[36]
            # if camera != relevant_camera:
            #     continue
            
            frame_idx = int(row[0])
            id = int(row[2])
            cls = row[3]
        
            if cls != '' and id not in obj_cls.keys():
                obj_cls[id] = cls
        
        
            # a. store 2D bbox
            if row[4] != '' and camera == relevant_camera:
                bbox_2d = np.array(row[4:8]).astype(int)
            else:
                bbox_2d = np.zeros(4)
                
            # b. store 3D bbox if it exists
            if row[11] != '' and camera == relevant_camera: # there was an initial 3D bbox prediction
                if id not in obj_heights.keys():
                    try:
                        est_height = (float(row[15]) + float(row[17]) + float(row[19]) + float(row[21])) - (float(row[7]) + float(row[9]) + float(row[11]) + float(row[13])) 
                        obj_heights[id] = est_height

                    except:
                        pass
                
                try:
                    bbox_3d = np.array(row[11:27]).astype(float).reshape(8,2)
                except:
                    print("error")
                    bbox_3d = np.zeros([8,2])

            else:
                bbox_3d = np.zeros([8,2])
                
            # c. store projected footprint coords if they exist
            if row[27] != '':
                footprint = np.array(row[27:35]).astype(float).reshape(4,2)
                
                # reproject these coords with inverse homography
                
                lmcs_bbox = np.array([footprint[:,0],footprint[:,1],[1,1,1,1]])#.transpose()
                
                out = np.matmul(H,lmcs_bbox)
                im_footprint = np.zeros([2,4])
                im_footprint[0,:] = out[0,:] / out[2,:]
                im_footprint[1,:] = out[1,:] / out[2,:]
                im_footprint = im_footprint.transpose()
            
            else:
                im_footprint = np.zeros([4,2])
            
            # d. store reprojected LMCS state as 3D box points if state exists
            if row[39] != '':
                # add image space box points
                x_center = float(row[39]) * 3.281
                y_center = float(row[40]) * 3.281
                theta    = float(row[41])
                width    = float(row[42])    * 3.281
                length   = float(row[43])   * 3.281
                
                fx = x_center + length * np.cos(theta) 
                ly = y_center + width/2 * np.cos(theta)
                bx = x_center
                ry = y_center - width/2 * np.cos(theta)    
                
                lmcs_bbox = np.array([[fx,fx,bx,bx],[ry,ly,ry,ly],[1,1,1,1]])#.transpose()
                
                out = np.matmul(H,lmcs_bbox)
                re_footprint = np.zeros([2,4])
                re_footprint[0,:] = out[0,:] / out[2,:]
                re_footprint[1,:] = out[1,:] / out[2,:]
                re_footprint = re_footprint.transpose()
                
                im_top = re_footprint.copy()
                try:
                    est_height = obj_heights[id]
                except KeyError:
                    est_height = 50
                    
                im_top[:,1] -= est_height/2.0
                rectified_bbox_3d = np.array(list(re_footprint.reshape(-1)) + list(im_top.reshape(-1)) )
                rectified_bbox_3d = rectified_bbox_3d.reshape(8,2)
            
            else:
                rectified_bbox_3d = np.zeros([8,2])
                    
                    
                
            if frame_idx in all_frame_data.keys():
                all_frame_data[frame_idx].append([id,bbox_2d,bbox_3d,lmcs_bbox,rectified_bbox_3d])
            else:
                all_frame_data[frame_idx] = [[id,bbox_2d,bbox_3d,lmcs_bbox,rectified_bbox_3d]]
                                
            
    # All data gathered from CSV
            
    # outfile = label_file.split("_track_outputs_3D.csv")[0] + "_3D.mp4"
    # out = cv2.VideoWriter(outfile,cv2.VideoWriter_fourcc(*'mp4v'), framerate, (3840,2160))
    
    cap= cv2.VideoCapture(sequence)
    ret,frame = cap.read()
    frame_idx = 0
    
    while ret:        
        
        if frame_idx in all_frame_data.keys():
            for box in all_frame_data[frame_idx]:
                
                # plot each box
                id              = box[0]
                bbox_2d         = box[1]
                bbox_3d         = box[2]
                bbox_lmcs       = box[3]
                bbox_rectified  = box[4]
                
                try:
                    cls = obj_cls[id]
                except:
                    cls = "sedan"
                
                
                
                DRAW = [[0,1,1,0,1,0,0,0], #bfl
                [0,0,0,1,0,1,0,0], #bfr
                [0,0,0,1,0,0,1,1], #bbl
                [0,0,0,0,0,0,1,1], #bbr
                [0,0,0,0,0,1,1,0], #tfl
                [0,0,0,0,0,0,0,1], #tfr
                [0,0,0,0,0,0,0,1], #tbl
                [0,0,0,0,0,0,0,0]] #tbr
        
                DRAW_BASE = [[0,1,1,1], #bfl
                        [0,0,1,1], #bfr
                        [0,0,0,1], #bbl
                        [0,0,0,0]] #bbr
                        
    
                color = class_colors[classes[cls]]
        
        
                color = (0,0,255)
                
                if show_2d:
                    frame = cv2.rectangle(frame,(int(bbox_2d[0]),int(bbox_2d[1])),(int(bbox_2d[2]),int(bbox_2d[3])),color,1)
                    
                color = (0,255,255)
                if show_3d:
                    for a in range(len(bbox_3d)):
                        ab = bbox_3d[a]
                        for b in range(a,len(bbox_3d)):
                            bb = bbox_3d[b]
                            if DRAW[a][b] == 1:
                                frame = cv2.line(frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,1)
                color = (0,255,0)             
                if show_LMCS:
                    for a in range(len(bbox_lmcs)):
                        ab = bbox_lmcs[a]
                        for b in range(len(bbox_lmcs)):
                            bb = bbox_lmcs[b]   
                        if DRAW_BASE[a][b] == 1:
                            frame = cv2.line(frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,2)
                
                color = (255,0,0)
                if show_rectified:
                    for a in range(len(bbox_rectified)):
                        ab = bbox_rectified[a]
                        for b in range(a,len(bbox_rectified)):
                            bb = bbox_3d[b]
                            if DRAW[a][b] == 1:
                                frame = cv2.line(frame,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),color,1)
                                
                
                label = "{} {}".format(cls,id)
                left = bbox_2d[0]
                top  = bbox_2d[1]
                frame = cv2.putText(frame,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),3)
                frame = cv2.putText(frame,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)

          
        #frame = cv2.resize(frame,(1920,1080))
        
        cv2.imshow("frame",frame)
        key = cv2.waitKey(int(1000/float(frame_rate)))
        if key == ord('q') or frame_idx > 1800:
            break
        
        # get next frame
        ret,frame = cap.read()
        frame_idx += 1     
        
    cv2.destroyAllWindows()
    cap.release()
    


if __name__ == "__main__":
    csv_file = "/home/worklab/Data/dataset_alpha_pretrials/rectified_all_img_re_rearranged.csv"
    sequence = "/home/worklab/Data/cv/video/ground_truth_video_06162021/trimmed/p1c5_00000.mp4"
    
    # csv_file = "/home/worklab/Data/dataset_alpha/automatic_3d/p1c2_3_track_outputs_3D.csv"
    # sequence = "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments/p1c2_3.mp4"
    
    
    plot_rectified_objects(sequence,csv_file,frame_rate = 10.0,show_2d = True,show_3d = True,show_LMCS = True ,show_rectified = True,save = False)