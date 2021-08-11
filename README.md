


# manual-track-labeler
This repository contains code for manually correcting 3D object tracking data to create "perfect" datasets for training and validation. It also contains code for associating instances of the same object across multiple cameras, videos or fields of view. Additional code is included for generating initial 2D object tracking outputs using *Localization-based Tracking* (LBT).

## How to use
If you haven't used the labeler tool before, start with [this video](nowhere). It provides an overview of installation, file management, and the labeling process.

## Requirements
- Python 3 ( Python 3.6 or later is best)
- pip or anaconda

## Installation
1. **Clone this repository**:
```
git clone https://github.com/DerekGloudemans/manual-track-labeler.git
```

2a. ** Install virtual python environment using `pip`**:

```
cd <path to manual-track-labeler>
python -m venv <path to manual-track-labeler>
source <path to manual-track-labeler>/bin/activate
pip install -r requirements.txt
```

Then, any time you open a new terminal and cd to the repo, you can reactivate the virtual environment with:

```
source <path to manual-track-labeler>/bin/activate
```

2b. **Using `conda` instead**:

```
cd <path to manual-track-labeler>
conda env create -f environment.yml
conda activate mtl-env
```

# Usage
First, copy or move all files you'd like to correct to `DATA/video/` and `DATA/labels/`. Be sure to "check the files out" on the [drive link](https://docs.google.com/spreadsheets/d/1BQwHaPUAT2V6C-czkjbrOMlsg5bcyRzcz5L_1_Blavk/edit?usp=sharing) so that no one else edits the same files at the same time.

```
cd annotate
python manual_annotator_3D.py <camera_name> <sequence_idx>
```

where `<camera_name>` is p1c1 or similar, and `<sequence_idx>` is an integer between 0 and 3 indicating the integer in the desired video file name.

## 3D Annotation Controls
 - `9` - advance to next frame
- `8` - return to previous frame. Note that only 200 frames are kept in buffer so attempting to move backwards more than 200 frames will cause an error
- `u` - undo the last labeling change made
- `r` - enter REASSIGN mode - in this mode, if you click on an object, enter an ID number, and press enter, the object will be assigned the new ID number in this and all subsequent frames. This is used when two object IDs correspond to the same object at different points in time
- `d` - enter DELETE mode - in this mode, if you click on an object and press enter, the object is deleted in this and all subsequent frames. Alternatively, you can click an object and enter a number, then press enter, to delete the object in that number of subsequent frames. 
- `a` - enter ADD mode - in this mode, if you click on an object, enter an ID number, and press enter, a new box is created and assigned to the relevant object. You can press enter to use the most recent ID number again, Note that if no other boxes exist for this object, you must also enter an object class and then press enter again. You will have to manually adjust the box until it fits the object.
- `w` - enter REDRAW mode - in this mode, you can click on a side of the box and drag it to change the box's dimensions. Note that at a given time, you can only drag the box along one of the 3 principal axes. You can toggle which axis is active by pressing `1`.
- `s` - pressing `s` followed by a frame number and enter will skip to that frame.Note that you will not be able to move backwards to previous frames after skipping because the previous frames will not have been buffered. Useful for skipping to where you left off on a previous labeling run.
- `f` - fast forward through frames. At any time, you can press enter or f to stop fast-forwarding.
- `q` - save and quit
- `i` - pressing `i` followed by an object ID number and enter will interpolate all missing bounding boxes for that object between the first and last frames in which it is detected. When adding boxes for an object, you can generally add a box every 5 frames and then interpolate the boxes between these.
- `m` - enter MOVE mode - in this mode, you can click within an object bounding box and drag to shift the whole object by that offset.
- right click twice within a box - realigns the object edges with the vanishing points, should be done after significantly moving or modifying a box.
- `k` - enter KEYFRAME mode - whenever you enter keyframe mode, you select an easily recognizable point on an object. Then, you subsequently select that point on the same object in different frames, and the box from the first frame is offset in the current frame according to the selected point. This is useful because you can fit one 3D box to an object well, and then copy and shift it to several subsequent frames before having to redraw it.
 
# Ignore the rest, it is not relevant at this time!

# Hear ye Hear ye (labelers skip to next section please)
If it shall be unto you to use this repository, know ye this: the following column format must be used for all .csv files or code will cease to work. You break it, you buy it! All unused columns should be left blank.

0. Frame # (0-indexed)
1. 	Timestamp	
2. 	Obj ID (unique integer per object)
3. 	Object class
4. 	2D bbox xmin	
5. 	2D bbox ymin
6. 	2D bbox xmax
7. 	2D bbox ymax	
8. 	2D vel_x	
9. 	2D vel_y	
10. Generation method
11. fbrx	 (3D bbox corners in image coordinates)
12. fbry	
13. fblx	
14. fbly	
15. bbrx	
16. bbry	
17. bblx	
18. bbly	
19. ftrx	
20. ftry	
21. ftlx	
22. ftly	
23. btrx	
24. btry	
25. btlx	
26. btly	
27. fbr_x (in LMCS)	
28. fbr_y	
29. fbl_x	
30. fbl_y	
31. bbr_x	
32. bbr_y	
33. bbl_x	
34. bbl_y	
35. direction (-1 if inbound / WB, 1 if outbound / EB)
36. camera from which detection originated
37. acceleration	
38. speed	
39. x	
40. y	
41. theta	
42. width	
43. length	

## Included files
- `manual_annotator_2D.py` -  for correcting 2D object tracking data
- **`manual_annotator_3D.py`** -  for correcting 3D object tracking data
- **`manual_annotator_associator.py`** -  for associating objects across sequences
- `_data/` -directory where all data is stored
- `train_detector.py` -for training object detector for LBT
- `train_localizer.py` - for training object localizer for LBT
- `track_all.py` - for tracking 
- `tracker_fsld.py` - source code for Localization-based Tracking
- `config/` - contains Kalman Filter parameters, model parameters, etc. for generating initial object tracks
- `model/`  - contains object detector/localizer code (Retinanet is used)
- `util/`   - contains various utilities for object tracking
- `requirements.txt` - pip-style spec for python environment
- `environment.yml` - conda-style spec for python environment

Ok Ok I know I'm not perfect, there are tons of other files not mentioned included in this repo. I'll get to updating the readme later, hopefully within the week.


## Cross-sequence Association Controls
- `1`- move to previous frame in left sequence
- `2`- move to next frame in left sequence
- `3`- move to previous frame in right sequence
- `4`- move to next frame in right sequence
- Click and drag from an object in the left sequence to an object in the right sequence to reassign the class and ID number of the object in the left sequence to the object in the right sequence.
- Click on an object in either sequence, type a class name, and press enter to reassign the new class to that object ID across all frames in both sequences
- `u` - undo the last labeling change
- `q` - save and quit


## Labeling guidelines
The following guidelines are used for 2D label correction:
- Ensure that each object is labeled by the same unique object ID for every frame that it is fully or almost fully within the frame.
- Especially for trucks, remove all boxes for which the truck is partially out of frame and the bounding box is not sized correctly. (This can be easily accomplished by finding the first frame in which the truck is entirely in-frame, reassigning the truck with a new, unused object ID at this frame (such as 99999), returning to the first frame in which the truck is detected, deleting the object, fast-forwarding to the first frame in which the relabeled (and thus not deleted) object is found, and reassigning the original label to that object
- All boxes not corresponding to an object should be deleted
- For cameras on the edge of each pole (cameras 1 and 6) objects should be labeled until they are at least 2/3 out of the field of view but don't need to be labeled through the entire frame.

A tutorial video can be found [here](https://github.com/DerekGloudemans/manual-track-labeler/releases/download/v1.0/zoom_1.mp4).
