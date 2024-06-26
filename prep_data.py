"""
Author(s): Jonathan Burkow (burkowjo@msu.edu), Michigan State University
           Greg Holste (giholste@gmail.com),  UT Austin
Last Modified: 06/05/2023
Description: Converts rib fracture data to proper format + directory structure for YOLOv5 training.

Usage: python prep_data.py --data_dir path/to/root/data/directory \
                           --save_dir path/to/save/directory \
                           --train_anno <train_csv_name>.csv \
                           --val_anno <val_csv_name>.csv \
                           --test_anno <test_csv_name>.csv \
                           --symbolic \
                           --clear_folder
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def parse_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--train_anno', type=str, default='train_annotations.csv')
    parser.add_argument('--val_anno', type=str, default='val_annotations.csv')
    parser.add_argument('--test_anno', type=str, default='test_annotations.csv')
    parser.add_argument('--symbolic', action='store_true', default=False)
    parser.add_argument('--clear_folder', action='store_true', default=False)

    return parser.parse_args()


def create_directories(out_dir: str, clear: bool = False) -> None:
    """
    Create directories for images and labels to be used for YOLOv5 training.

    Parameters
    ----------
    out_dir : path to root directory for all YOLOv5 data folders
    """
    if os.path.isdir(out_dir) and clear:
        shutil.rmtree(out_dir)
        print('Removed previous data directories.')
    os.mkdir(out_dir)
    os.mkdir(os.path.join(out_dir, 'images'))
    os.mkdir(os.path.join(out_dir, 'images', 'train'))
    os.mkdir(os.path.join(out_dir, 'images', 'val'))
    os.mkdir(os.path.join(out_dir, 'images', 'test'))
    os.mkdir(os.path.join(out_dir, 'labels'))
    os.mkdir(os.path.join(out_dir, 'labels', 'train'))
    os.mkdir(os.path.join(out_dir, 'labels', 'val'))
    os.mkdir(os.path.join(out_dir, 'labels', 'test'))
    print('Created all data directories.')


def copy_images(annot_path: str, set_name: str, save_dir: str, symbolic: bool = False) -> None:
    """
    Copy images from the main data directory to the YOLO directory structure; can also be symbolic
    links if file count or disk space needs to be considered.

    Parameters
    ----------
    annot_path : path to annotation CSV to copy images from
    set_name   : whether the annotation CSV is of training, validation, or test set images
    symbolic   : true/false on whether to hard copy images or to use symbolic links
    """
    annot = pd.read_csv(annot_path, names=['img_path', 'x1', 'y1', 'x2', 'y2', 'label'])

    for _, img_path in tqdm(enumerate(annot['img_path'].unique().tolist()), desc=f'Copying {set_name.title()} Images', total=len(annot['img_path'].unique().tolist())):
        if img_path.endswith('\''):  # manually overriding error where there would be an extra single quotation mark in an image ID
            img_path = img_path[:-1]  # remove last character

        patient_id = img_path.split('/')[-1]

        if symbolic:
            os.symlink(img_path, os.path.join(save_dir, 'images', set_name, patient_id))
        else:
            shutil.copy(img_path, os.path.join(save_dir, 'images', set_name))


def convert_to_yolo(annot_path: str, set_name: str, save_dir: str) -> None:
    """
    Convert labels from RetinaNet CSV format into YOLO txt format.

    Parameters
    ----------
    annot_path : path to annotation CSV to copy images from
    set_name   : whether the annotation CSV is of training, validation, or test set images
    """
    annot = pd.read_csv(annot_path, names=['img_path', 'x1', 'y1', 'x2', 'y2', 'label'])

    classes = ['fracture']

    for _, img_path in tqdm(enumerate(annot['img_path'].unique().tolist()), desc=f'Converting {set_name.title()} Labels', total=len(annot['img_path'].unique().tolist())):
        patient_id = img_path.split('/')[-1].split('.')[0]
        if not os.path.isfile(img_path):
            print(f"IMAGE FOR {patient_id} NOT FOUND/NOT READ IN")
            continue

        sub_df = annot[annot['img_path'] == img_path]

        # Check for any pd.NA values; skip if they exist (Don't need label files for them)
        if sub_df.isnull().values.any():
            continue

        with open(os.path.join(save_dir, 'labels', set_name, f"{patient_id}.txt"), 'w') as out:
            if img_path.endswith('\''):
                img_path = img_path[:-1]  # remove last character

            img = cv2.imread(img_path, 0)

            img_h, img_w = img.shape[0], img.shape[1]

            for _, row in sub_df.iterrows():
                # Get items for YOLO format
                cls_idx = classes.index(row['label'])

                x_c = int((row['x1'] + row['x2']) / 2)
                y_c = int((row['y1'] + row['y2']) / 2)
                w = row['x2'] - row['x1']
                h = row['y2'] - row['y1']

                data = [cls_idx, x_c / img_w, y_c / img_h, w / img_w, h / img_h]

                out.write(' '.join(str(s) for s in data) + '\n')


def main():
    """Main Function"""
    parse_args = parse_cmd_args()

    create_directories(parse_args.save_dir, clear=parse_args.clear_folder)

    copy_images(os.path.join(parse_args.data_dir, parse_args.train_anno), 'train', parse_args.save_dir, parse_args.symbolic)
    copy_images(os.path.join(parse_args.data_dir, parse_args.val_anno), 'val', parse_args.save_dir, parse_args.symbolic)
    copy_images(os.path.join(parse_args.data_dir, parse_args.test_anno), 'test', parse_args.save_dir, parse_args.symbolic)

    convert_to_yolo(os.path.join(parse_args.data_dir, parse_args.train_anno), 'train', parse_args.save_dir)
    convert_to_yolo(os.path.join(parse_args.data_dir, parse_args.val_anno), 'val', parse_args.save_dir)
    convert_to_yolo(os.path.join(parse_args.data_dir, parse_args.test_anno), 'test', parse_args.save_dir)


if __name__ == "__main__":
    print(f"\n{'Starting execution: ' + Path(__file__).name:-^80}\n")
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print(f"\n{'Done!':-^80}")
    print(f'Execution finished in {elapsed:.3f} seconds ({time.strftime("%-H hr, %-M min, %-S sec", time.gmtime(elapsed))}).\n')
