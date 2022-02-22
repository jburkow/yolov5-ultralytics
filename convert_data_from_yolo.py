"""
Filename: convert_data_from_yolo.py
Author(s): Jonathan Burkow (burkowjo@msu.edu), Michigan State University
Last Modified: 02/22/2022
Description: Converts detection information from YOLO format into RetinaNet format for analysis.
"""

import argparse
import os
import time
from typing import List, Tuple

import cv2
import pandas as pd
from tqdm import tqdm


def xywh_to_xyxy(x: List[float], width: int = 640, height: int = 640) -> Tuple[int, int, int, int]:
    """
    Converts a YOLO model prediction from [x, y, w, h] format to [x1, y1, x2, y2].

    Parameters
    ----------
    x      : list containing YOLO bounding box prediction; [x, y, w, h]
    width  : width of the image
    height : height of the image
    """
    x1 = width * (x[0] - x[2] / 2)
    y1 = height * (x[1] - x[3] / 2)
    x2 = width * (x[0] + x[2] / 2)
    y2 = height * (x[1] + x[3] / 2)
    return int(x1), int(y1), int(x2), int(y2)


def main(parse_args):
    """Main Function"""
    # Get list of all images in test dataset and images with detections
    test_dataset = [os.path.join(root, file) for root, _, files in os.walk(parse_args.test_image_dir) for file in files]
    detect_dataset = [os.path.join(root, file) for root, _, files in os.walk(parse_args.labels_dir) for file in files]

    # Add all bounding boxes to a single DataFrame
    detection_df = pd.DataFrame(columns=['ID','x1', 'y1', 'x2', 'y2', 'Prob'])
    for _, test_image_path in tqdm(enumerate(test_dataset), desc='Converting Labels from YOLO', total=len(test_dataset)):
        patient_id = test_image_path.split('/')[-1].split('.')[0]
        if patient_id not in '\t'.join(detect_dataset):  # Add a row with empty values if matching PatientID not found
            detection_df = detection_df.append({'ID' : os.path.join(parse_args.img_path, f"{patient_id}.png"),
                                                'x1' : '', 'y1' : '', 'x2' : '', 'y2' : '', 'Prob' : ''}, ignore_index=True)
            continue

        # Get corresponding index in detection data list of the current test set image
        index = [idx for idx, s in enumerate(detect_dataset) if patient_id in s][0]

        test_image = cv2.imread(test_image_path)
        test_image_shape = test_image.shape

        im_preds = pd.read_csv(detect_dataset[index], delim_whitespace=True, names=['class', 'x', 'y', 'w', 'h', 'score'])

        for _, row in im_preds.iterrows():
            xyxy = xywh_to_xyxy([row['x'], row['y'], row['w'], row['h']], test_image_shape[1], test_image_shape[0])
            detection_df = detection_df.append({'ID' : os.path.join(parse_args.img_path, f"{patient_id}.png"),
                                                'x1' : xyxy[0], 'y1' : xyxy[1], 'x2' : xyxy[2], 'y2' : xyxy[3], 'Prob' : row['score']}, ignore_index=True)

    detection_df.sort_values(['ID', 'Prob'], inplace=True, ignore_index=True, ascending=[True, False])
    detection_df.to_csv(os.path.join(parse_args.save_dir, parse_args.save_name), header=False, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_image_dir', type=str, required=True,
                        help='Path to directory containing all of the test set images.')

    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Path to directory containing all output label files from YOLO.')

    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to directory containing all processed images to add in before saving CSV.')

    parser.add_argument('--save_dir', type=str, default='./',
                        help='Path to directory to save CSV to.')

    parser.add_argument('--save_name', type=str, default='yolo_model_predictions.csv',
                        help='Name of the CSV file to save as.')

    args = parser.parse_args()

    print('\nStarting execution...')
    start_time = time.perf_counter()
    main(args)
    end_time = time.perf_counter()
    print('Done!')
    print(f'Execution finished in {end_time - start_time:.3f} seconds.\n')
