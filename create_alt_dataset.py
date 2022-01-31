"""
Author: Greg Holste
Last Modified: 06/23/21
Description: Script to create YOLO-formatted versions of rib fracture dataset with other preprocessing methods (a, b, or c -- see argparse help). Must run prep_data.py first.

Usage: python create_alt_dataset.py \
    --method b \
    --data_dir /mnt/research/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210420/ \
    --seg_dir /mnt/research/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210420/segmentation_masks/cropped_npy/ \
    --source_dir /mnt/home/holstegr/MIDI/RibFracDet/YOLOv5-RibFrac/ribfrac_051721/
"""

import cv2
import numpy as np
import os
import shutil
import imageio
import argparse
import pandas as pd
import tqdm

def process_images(annot_path, seg_dir, method='a'):
    annot = pd.read_csv(annot_path, names=['img_path', 'x1', 'y1', 'x2', 'y2', 'label'])

    set_name = annot_path.split('/')[-1].split('_')[0]

    if method == 'a':

        for img_path in tqdm.tqdm(list(set(annot['img_path']))):
            patient_id = img_path.split('/')[-1].split('.')[0]

            seg_path = os.path.join(seg_dir, [f for f in os.listdir(seg_dir) if patient_id in f][0])

            # Load image and segmentation
            img = cv2.imread(img_path)
            seg = np.load(seg_path)

            fg = ((seg.sum(axis=-1) - seg[:, :, 0])*255).astype(np.uint8)

            # Create rib segmentation
            masked_img = cv2.bitwise_and(img[:, :, 0], img[:, :, 0], mask=fg)

            rib_seg = cv2.adaptiveThreshold(masked_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 0)

            out = np.stack([masked_img, masked_img, rib_seg], axis=-1)

            imageio.imwrite(os.path.join(out_dir, 'images', set_name, patient_id + '.png'), out.astype(np.uint8))

    elif method == 'b':

        for img_path in tqdm.tqdm(list(set(annot['img_path']))):
            patient_id = img_path.split('/')[-1].split('.')[0]

            seg_path = os.path.join(seg_dir, [f for f in os.listdir(seg_dir) if patient_id in f][0])

            # Load image and segmentation
            img = cv2.imread(img_path)
            seg = np.load(seg_path)

            fg = ((seg.sum(axis=-1) - seg[:, :, 0])*255).astype(np.uint8)

            # Create rib segmentation
            masked_img = cv2.bitwise_and(img[:, :, 0], img[:, :, 0], mask=fg)

            rib_seg = cv2.adaptiveThreshold(masked_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 0)

            out = np.stack([rib_seg, rib_seg, rib_seg], axis=-1)

            imageio.imwrite(os.path.join(out_dir, 'images', set_name, patient_id + '.png'), out.astype(np.uint8))

    elif method == 'c':

        raw_img_paths = [s.replace('cropped_histeq_png', 'cropped_png') for s in list(set(annot['img_path']))]

        for img_path in tqdm.tqdm(raw_img_paths):
            patient_id = img_path.split('/')[-1].split('.')[0]

            seg_path = os.path.join(seg_dir, [f for f in os.listdir(seg_dir) if patient_id in f][0])

            # Load image and segmentation
            img = cv2.imread(img_path)
            seg = np.load(seg_path)

            fg = ((seg.sum(axis=-1) - seg[:, :, 0])*255).astype(np.uint8)

            # Create rib segmentation
            masked_img = cv2.bitwise_and(img[:, :, 0], img[:, :, 0], mask=fg)

            masked_histeq_img = cv2.equalizeHist(masked_img)
            masked_bilateral_img = cv2.bilateralFilter(masked_img, 9, 75, 75)

            out = np.stack([masked_img, masked_histeq_img, masked_bilateral_img], axis=-1)

            imageio.imwrite(os.path.join(out_dir, 'images', set_name, patient_id + '.png'), out.astype(np.uint8))

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='a',
    help='One of [a, b, c]. a = [fg-masked image, fg-masked image, rib seg], b = [fg-masked raw image, fg-masked hist eq image, fg-masked bilateral filtered image].')
parser.add_argument('--data_dir', type=str, default='/mnt/research/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210420/',
    help='Path to directory containing rib fracture images and labels.')
parser.add_argument('--seg_dir', type=str, default='/mnt/research/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210420/segmentation_masks/cropped_npy/',
    help='Path to directory containing chest segmentations associated with rib fracture images.')
parser.add_argument('--source_dir', type=str, default='/mnt/home/holstegr/MIDI/RibFracDet/YOLOv5-RibFrac/ribfrac_051721/',
    help='Path to directory containing YOLO-formatted data from which to copy labels.')
args = parser.parse_args()

assert args.method in ['a', 'b', 'c'], "--method must be one of ['a', 'b']" 

if args.method == 'a':
    out_dir = 'ribfrac_ri-ri-rs_061021'
elif args.method == 'b':
    out_dir = 'ribfrac_rs-rs-rs_061021'
elif args.method == 'c':
    out_dir = 'ribfrac_raw-histeq-bilateral_061021'

if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
os.mkdir(os.path.join(out_dir, 'images'))
os.mkdir(os.path.join(out_dir, 'images', 'train'))
os.mkdir(os.path.join(out_dir, 'images', 'val'))
os.mkdir(os.path.join(out_dir, 'images', 'test'))

# Copy labels
shutil.copytree(os.path.join(args.source_dir, 'labels/'), os.path.join(out_dir, 'labels/'))

# Remove cached files if they exist
rm_files = [f for f in os.listdir(os.path.join(out_dir, 'labels')) if 'cache' in f]
if len(rm_files) != 0:
    for f in rm_files:
        os.remove(os.path.join(out_dir, 'labels', f))

# Process images
process_images(os.path.join(args.data_dir, 'train_annotations.csv'), args.seg_dir, method=args.method)
process_images(os.path.join(args.data_dir, 'val_annotations.csv'), args.seg_dir, method=args.method)
process_images(os.path.join(args.data_dir, 'test_annotations.csv'), args.seg_dir, method=args.method)