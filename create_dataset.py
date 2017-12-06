#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os
import random as rand
import sys
import cascade_classifier as cc

DATASET_DIR = "dataset/"
TRAIN_CSV_NAME = 'train.csv'
TEST_CSV_NAME = 'test.csv'
TRAIN_DATA = []
TEST_DATA = []


def write_csv(outdir, csvname, data):
    writer = csv.writer(open(os.path.join(outdir, csvname), 'w'), lineterminator='\n')
    writer.writerows(data)


def addDataset(dataPaths, label):
    for dataPath in dataPaths:
        if os.path.exists(dataPath):
            if 50 > rand.randrange(100):
                TRAIN_DATA.append([dataPath, label])
            else:
                TEST_DATA.append([dataPath, label])
        else:
            print("ERROR {} not found.".format(dataPath))


if __name__ == '__main__':
    rand.seed()

    if not os.path.isdir(DATASET_DIR):
        sys.exit('%s is not directory' % DATASET_DIR)

    labels = {
        "slim": 0,
        "medium": 1,
        "heavy": 2,
        "slimMacho": 3,
        "macho": 4,
        "superMacho": 5,
        "athlete": 6,
        "physique": 7,
        "bodybuild": 8,
        "sumoWrestling": 9,
    }
    exts = ['.PNG', '.JPG', '.JPEG']

    for dirpath, dirnames, filenames in os.walk(DATASET_DIR):
        for dirname in dirnames:
            if dirname in labels:
                labelnumber = labels[dirname]
                member_dir = os.path.join(dirpath, dirname)
                for dirpath2, dirnames2, filenames2 in os.walk(member_dir):
                    if not dirpath2.endswith(dirname):
                        continue
                    for filename2 in filenames2:
                        (fn, ext) = os.path.splitext(filename2)
                        if ext.upper() in exts:
                            image_path = os.path.join(dirpath2, filename2)
                            ext_image_paths = cc.detectObjectFromImage(image_path, cc.UPPER_BODY)
                            addDataset(ext_image_paths, labelnumber)
                print("{} Done.".format(member_dir))

    write_csv(DATASET_DIR, TRAIN_CSV_NAME, TRAIN_DATA)
    write_csv(DATASET_DIR, TEST_CSV_NAME, TEST_DATA)
