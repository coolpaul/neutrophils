# python 3.8 yolo11

import os
from ultralytics import YOLO
import numpy as np
import glob
import pandas as pd
import cv2
from datetime import datetime
from tqdm import tqdm

from patched_yolo_infer import MakeCropsDetectThem, CombineDetections

def convert_boxes_2_yolo(boxes=None, img_size=(256, 256)):
    yolo_boxes = []
    width = img_size[0]
    height = img_size[1]
    for box in boxes:
        x_top, y_top, x_bottom, y_bottom = box
        xc = (x_top + x_bottom) / 2 / width
        yc = (y_top + y_bottom) / 2 / height
        w = (x_bottom - x_top) / width
        h = (y_bottom - y_top) / height
        yolo_boxes.append([xc, yc, w, h])
    return yolo_boxes

def write_yolo_boxes_2_df(yolo_boxes=None, ids=None, confidences=None, image_file_name=None):
    dir_name = os.path.dirname(image_file_name)
    file_name = os.path.basename(image_file_name)
    yolo_file_name = os.path.join(dir_name, file_name.replace('.jpg', '.txt'))
    # create dataframe from lists
    dictionary = {'id': ids, 'yolo': yolo_boxes, 'conf': confidences, 'file_name': file_name}
    df = pd.DataFrame(dictionary)
    return df

def define_empty_dataframe():
    # define empty data frame with columns
    return pd.DataFrame(columns=['id', 'yolo', 'conf', 'file_name'])

def draw_boxes_on_image(image, detections, labels=None):
    height, width = image.shape[:2]
    for det in detections:
        xc, yc, w, h = det['xc'], det['yc'], det['w'], det['h']
        x_center = xc * width
        y_center = yc * height
        box_w = w * width
        box_h = h * height

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Optional label
        if labels is not None:
            label_text = labels.get(det['id'], 'Object')
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':
    # set parameter values
    confidence = 0.3
    crop_size = 512
    overlap = 30  # percentage overlap
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H-%M-%S")
    # yolo model should be named yolo11.pt to comply with patched_yolo_infer
    yolo_model_path = 'yolo11.pt' # copy of the best.pt in the training/stage_2_model/weights/ directory
    histo_images_dir = 'test_cases'
    
    df = define_empty_dataframe()
    # Process images
    for i in tqdm(os.listdir(histo_images_dir)):
        print(f'Processing: {i}')
        image_path = os.path.join(histo_images_dir, i)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {i}")
            continue
        height, width, channels = image.shape

        # Get crops and detections
        element_crops = MakeCropsDetectThem(
            image=image,
            model_path=yolo_model_path,
            segment=False,
            show_crops=False,
            shape_x=crop_size,
            shape_y=crop_size,
            overlap_x=overlap,
            overlap_y=overlap,
            conf=confidence,
            iou=0.2,
            resize_initial_size=True
        )
        result = CombineDetections(element_crops, nms_threshold=0.001, match_metric='IOU')

        # Extract detection info
        confidences = result.filtered_confidences
        boxes = result.filtered_boxes
        classes_ids = result.filtered_classes_id

        # Convert boxes to YOLO format
        boxes_yolo = convert_boxes_2_yolo(boxes=boxes, img_size=(width, height))
        
        # Create a DataFrame for current image detections
        df_add = write_yolo_boxes_2_df(
            yolo_boxes=boxes_yolo,
            ids=classes_ids,
            confidences=confidences,
            image_file_name=i
        )

        # Append current detections to the main DataFrame
        df = pd.concat([df, df_add], axis=0)

        # Expand yolo list column into separate columns for visualization
        df[['xc', 'yc', 'w', 'h']] = pd.DataFrame(df['yolo'].tolist(), index=df.index)

        # Collect detections for current image for visualization
        detections = []
        for _, row in df[df['file_name'] == i].iterrows():
            detections.append({
                'xc': row['xc'],
                'yc': row['yc'],
                'w': row['w'],
                'h': row['h'],
                'id': row['id']
            })

        # Draw detections on the image
        draw_boxes_on_image(image, detections)

        # Display the image with detections
        cv2.imshow(f'Detections for {i}', image)
        cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

# Save final DataFrame to CSV
df.to_csv(f'result_{now_str}.csv', index=False)











