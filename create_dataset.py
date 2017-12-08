#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os
import random as rand
import sys
import cascade_classifier as cc
import cv2

DATASET_DIR = "dataset/"
TRAIN_CSV_NAME = 'train.csv'
TEST_CSV_NAME = 'test.csv'
TRAIN_LIST = []
TEST_LIST = []


def write_csv(outdir, csvname, data):
    writer = csv.writer(open(os.path.join(outdir, csvname), 'w'), lineterminator='\n')
    writer.writerows(data)


def addDataset(dataPaths, label):
    for dataPath in dataPaths:
        if os.path.exists(dataPath):
            if 50 > rand.randrange(100):
                TRAIN_LIST.append([dataPath, label])
            else:
                TEST_LIST.append([dataPath, label])
        else:
            print("ERROR {} not found.".format(dataPath))


def createLabelList(list):
    '''
    ラベルのリストを作成して返す
    :param list: "文字列”
    :return: label_list [[str, int]]
    '''
    label_num = 0
    label_list = []
    for label_name in list:
        label_list.append([label_name, label_num])
        label_num += 1
    return label_list


def resizeImageAndSave(image_path, resize):
    PREFIX_RESIZE = "_resize"
    path = os.path.splitext(image_path)
    new_image_path = path[0] + PREFIX_RESIZE + path[1]
    if PREFIX_RESIZE in image_path or cc.PREFIX_EXTRACTED in image_path:
        new_image_path = image_path
    else:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (resize, resize))
        cv2.imwrite(new_image_path, image)
    return new_image_path


if __name__ == '__main__':
    rand.seed()

    if not os.path.isdir(DATASET_DIR):
        sys.exit('%s is not directory' % DATASET_DIR)

    exts = ['.PNG', '.JPG', '.JPEG']
    label_list = []
    for dirpath, dirnames, filenames in os.walk(DATASET_DIR):
        if dirnames != []:
            label_list = createLabelList(dirnames)
            write_csv(DATASET_DIR, "labels.csv", label_list)

    # label[0]: dirname, label[1]: number
    for label in label_list:
        dirname = label[0]
        member_dir = os.path.join(DATASET_DIR, dirname)
        for dirpath, dirnames, filenames in os.walk(member_dir):
            if not dirpath.endswith(dirname):
                continue
            for filename in filenames:
                (fn, ext) = os.path.splitext(filename)
                if ext.upper() in exts:
                    image_path = os.path.join(dirpath, filename)

                    # save_image_paths = cc.detectObjectFromImage(image_path, cc.FULL_BODY)
                    # addDataset(save_image_paths, label[1])

                    # save_image_paths = resizeImageAndSave(image_path, 100)
                    # addDataset([save_image_paths], label[1])

                    save_image_paths = cc.detect_contour(image_path)
                    addDataset(save_image_paths, label[1])
        print("{} Done.".format(member_dir))

    write_csv(DATASET_DIR, TRAIN_CSV_NAME, TRAIN_LIST)
    write_csv(DATASET_DIR, TEST_CSV_NAME, TEST_LIST)
