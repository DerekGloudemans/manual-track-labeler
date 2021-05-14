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

## Controls


## Labeling guidelines

