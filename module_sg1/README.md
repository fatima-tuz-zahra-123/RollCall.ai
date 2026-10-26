# YOLOv12n Face

This folder contains the YOLOv12 nano face checkpoint, WIDER FACE image
splits, training/validation labels, and a small command-line runner.

## Environment

Use Python 3.10 or newer, then install dependencies from this folder:

```powershell
python -m pip install -r requirements.txt
```

## Prepare labels

The WIDER FACE archive publishes annotations for train and validation, not
test. Run this once to convert the included official annotations into YOLO
labels:

```powershell
python run.py prepare
```

## Train

The default trains from the included pretrained YOLOv12n-face checkpoint:

```powershell
python run.py train --epochs 100 --imgsz 640 --batch 16
```

Training outputs are saved under `runs/train/`. Use `--device 0` for a CUDA
GPU or `--device cpu` to force CPU.

## Validate

Evaluate the included checkpoint on the WIDER validation split:

```powershell
python run.py val
```

To validate a trained checkpoint, pass its path:

```powershell
python run.py val --model runs/train/face/weights/best.pt
```

## Predict

Run the pretrained model on the example WIDER test image:

```powershell
python run.py predict
```

Or pass an image, folder, or video:

```powershell
python run.py predict --source datasets/WIDER_test/images/0--Parade/0_Parade_marchingband_1_1007.jpg
```

Annotated predictions are saved under `runs/predict/`. The WIDER test set has
no public ground-truth labels, so use the validation split for metric
evaluation.

## Dataset layout

* `datasets/WIDER_train/images` and `datasets/WIDER_val/images` are the
  original training and validation pictures.
* `datasets/WIDER_train/labels` and `datasets/WIDER_val/labels` are generated
  by `python run.py prepare`.
* `datasets/WIDER_test/images` contains the official test pictures. No test
  labels are included or generated.
* `datasets/wider_face_split/wider_face_split` contains the official WIDER
  train/validation box annotation text files.

The original WIDER folders outside this project were copied, not moved.
WIDER FACE is provided by its dataset authors; consult the dataset's terms
before redistribution.
