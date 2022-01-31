"""
Author: Greg Holste
Last Updated: 06/23/21
Description: Converts YOLOv5 predictions (output of yolov5/test.py or yolov5/detect.py) to format used for evaluation via https://github.com/jburkow/rib_fracture_utils/blob/master/compare_reads.py.

Usage: python convert_preds.py \
    --pred_dir <path to model detections>/labels/ \
    --gt_annot_path /mnt/research/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210420/test_annotations.csv
"""

import pandas as pd
import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', type=str, required=True,
    help='Path to directory containing predictions from a trained YOLOv5 model.')
parser.add_argument('--gt_annot_path', type=str, default='/mnt/research/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210420/test_annotations.csv',
    help='Path to ground truth annotations for the set (val or test) that predictions were obtained from.')
args = parser.parse_args()

test_annot = pd.read_csv(args.gt_annot_path, names=['img_path', 'x1', 'y1', 'x2', 'y2', 'label'])

img_paths = sorted(list(set(test_annot['img_path'])))

out_dict = {'img_path': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'score': []}
for img_path in img_paths:
    patient_id = img_path.split('/')[-1].split('.')[0]

    fname = os.path.join(args.pred_dir, patient_id + '.txt')

    if os.path.isfile(fname):
        img_preds = pd.read_csv(os.path.join(args.pred_dir, patient_id + '.txt'), delimiter=' ', names=['class', 'x_c', 'y_c', 'w', 'h', 'score'])
        
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[0], img.shape[1]

        for _, row in img_preds.iterrows():
            w = row['w']*img_w
            h = row['h']*img_h
            x_c_raw = row['x_c']*img_w
            y_c_raw = row['y_c']*img_h

            x1, x2 = int(x_c_raw-w/2), int(x_c_raw+w/2)
            y1, y2 = int(y_c_raw-h/2), int(y_c_raw+h/2)

            out_dict['img_path'].append(img_path)
            out_dict['x1'].append(x1)
            out_dict['y1'].append(y1)
            out_dict['x2'].append(x2)
            out_dict['y2'].append(y2)
            out_dict['score'].append(row['score'])
    else:
        out_dict['img_path'].append(img_path)
        out_dict['x1'].append(0)
        out_dict['y1'].append(0)
        out_dict['x2'].append(0)
        out_dict['y2'].append(0)
        out_dict['score'].append(0)

model_name = args.pred_dir.split('/')[-3]

save_dir = '/'.join(args.pred_dir.split('/')[:-2])

out_df = pd.DataFrame(out_dict)
out_df.to_csv(os.path.join(save_dir, 'preds.csv'), index=False, header=False)

