# manual-track-labeler
This repository contains code for manually correcting 2D object tracking data to create "perfect" datasets for training and validation. It also contains code for associating instances of the same object across multiple cameras, videos or fields of view. Additional code is included for generating initial 2D object tracking outputs using *Localization-based Tracking* (LBT).

## Included files
**`manual_annotator_2D.py`** -  for correcting 2D object tracking dat
**`manual_annotator_associator.py`** -  for associating objects across sequences
`train_detector.py` -for training object detector for LBT
`train_localizer.py` - for training object localizer for LBT
`track_all.py` - for tracking 
`tracker_fsld.py` - source code for Localization-based Tracking
`config/` - contains Kalman Filter parameters, model parameters, etc. for generating initial object tracks
`model/`  - contains object detector/localizer code (Retinanet is used)
`util/`   - contains various utilities for object tracking

# Installation


# Usage


## Labeling guidelines

## Controls
