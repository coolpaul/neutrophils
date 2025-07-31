# Neutrophils

yolo detection of neutrophils in histology slides

**Requirements**

| opencv-python==4.11.0.86  |
| ------------------------- |
| numpy==1.26.4             |
| pandas==2.2.3             |
| patched_yolo_infer==1.3.8 |
| tqdm==4.67.1              |
| ultralytics==8.3.109      |

The file yolo11.pt contains the model with trained weights for neutrophil detection and can be used with patched yolo inference as demonstrated in the Python script predict_neutro_histo_slide.py.
