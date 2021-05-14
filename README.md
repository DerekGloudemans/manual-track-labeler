# manual-track-labeler
This repository contains code for manually correcting 2D object tracking data to create "perfect" datasets for training and validation. It also contains code for associating instances of the same object across multiple cameras, videos or fields of view. Additional code is included for generating initial 2D object tracking outputs using *Localization-based Tracking* (LBT).

## Included files
- **`manual_annotator_2D.py`** -  for correcting 2D object tracking dat
- **`manual_annotator_associator.py`** -  for associating objects across sequences
- `outputs/` -directory where all data is stored
- `train_detector.py` -for training object detector for LBT
- `train_localizer.py` - for training object localizer for LBT
- `track_all.py` - for tracking 
- `tracker_fsld.py` - source code for Localization-based Tracking
- `config/` - contains Kalman Filter parameters, model parameters, etc. for generating initial object tracks
- `model/`  - contains object detector/localizer code (Retinanet is used)
- `util/`   - contains various utilities for object tracking
- `requirements.txt` - pip-style spec for python environment
- `environment.yml` - conda-style spec for python environment
- 
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
First, copy or move all files you'd like to correct to `outputs/track/`

```
python manual_annotator_2D.py <path>
```

where `<path>` is the name of the track output file stored in `outputs/track/`. The resulting corrected track file will be written to `outputs/track_corrected/`.

## 2D Annotation Controls
 - `9` - advance to next frame
- `8` - return to previous frame. Note that only 200 frames are kept in buffer so attempting to move backwards more than 200 frames will cause an error
- `u` - undo the last labeling change made
- `r` - enter REASSIGN mode - in this mode, if you click on an object, enter an ID number, and press enter, the object will be assigned the new ID number in this and all subsequent frames
- `d` - enter DELETE mode - in this mode, if you click on an object, the object is deleted in this and all subsequent frames
- `a` - enter ADD mode - in this mode, if you click and drag to draw a rectangular bounding box, enter an ID number, and press enter, a new box is created and assigned to the relevant object. You can press enter to use the most recent ID number again, if adding multiple boxes for the same object in multiple frames. Note that if no other boxes exist for this object, you must also enter an object class and then press enter again. Lastly, after a box is added, the frame is automatically advanced. You can return to the previous frame to inspect the drawn box.
- `w` - enter REDRAW mode - in this mode, if you click and drag to draw a box overlapping with an existing box, the new box will replace the old box, useful for correcting slightly innaccurate boxes
- `s` - pressing `s` followed by a frame number and enter will skip to that frame.Note that you will not be able to move backwards to previous frames after skipping because the previous frames will not have been buffered. Useful for skipping to where you left off on a previous labeling run.
- `f` - fast forward through frames. At any time, you can press enter or f to stop fast-forwarding.
- `q` - save and quit
- `i` - pressing `i` followed by an object ID number and enter will interpolate all missing bounding boxes for that object between the first and last frames in which it is detected. When adding boxes for an object, you can generally add a box every 5-10 frames and then interpolate the boxes between these.

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

