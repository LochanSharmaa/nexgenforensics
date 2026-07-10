# WIDER Face Training Setup

The WIDER Face archives have been extracted and converted for YOLO face detection.

## Prepared Files

- Dataset root: `backend/runtime/datasets/wider_face/raw`
- YOLO config: `backend/runtime/datasets/wider_face/prepared/wider_face_yolo.yaml`
- Manifest: `backend/runtime/datasets/wider_face/prepared/wider_face_manifest.csv`
- Report: `backend/runtime/datasets/wider_face/prepared/wider_face_report.json`
- Local training environment: `backend/runtime/venvs/wider-yolo`

## Verified Counts

- Train: 12,880 images, 156,994 valid face boxes
- Validation: 3,226 images, 39,112 valid face boxes
- Test: 16,097 images, no public test labels
- Missing images: 0
- Unreadable images: 0

## Rebuild Dataset

```powershell
& 'C:\Users\locha\Desktop\nexgenforensics\backend\runtime\venvs\wider-yolo\Scripts\python.exe' backend\scripts\prepare_wider_face.py `
  --output-root backend\runtime\datasets\wider_face `
  --split-zip 'C:\Users\locha\Downloads\wider_face_split.zip' `
  --train-zip 'C:\Users\locha\Downloads\WIDER_train.zip' `
  --val-zip 'C:\Users\locha\Downloads\WIDER_val (2).zip' `
  --test-zip 'C:\Users\locha\Downloads\WIDER_test.zip'
```

## Full Training

This machine currently runs PyTorch on CPU, so full training will be slow. Use a CUDA-enabled PyTorch environment and set `--device 0` if an NVIDIA GPU is available.

```powershell
& 'C:\Users\locha\Desktop\nexgenforensics\backend\runtime\venvs\wider-yolo\Scripts\python.exe' backend\scripts\train_wider_face_yolo.py `
  --model yolov8n.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 16 `
  --device cpu `
  --project backend\runtime\training `
  --name wider_face_yolo
```

## Smoke Test

A CPU smoke training run completed successfully with `yolov8n.yaml`, 1 epoch, image size 64, batch 1, and 0.001 training fraction.
