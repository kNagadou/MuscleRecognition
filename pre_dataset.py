#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os
import random
import sys
import cv2

# cascade path
CASCADE_PATH = "opencv_cascade_clasisifier/"

# cascade file
UPPER_BODY = "haarcascade_upperbody.xml"


def pre_work(image_path):
    # http://famirror.hateblo.jp/entry/2015/12/19/180000
    resize = 100
    path = os.path.splitext(image_path)
    new_image_path = path[0] + '_face' + path[1]

    # 処理済みであれば再処理しない
    if '_face' in image_path:
        return image_path

    if os.path.exists(new_image_path):
        print('already exists file. {}'.format(new_image_path))
        return ""

    # ファイル読み込み
    image = cv2.imread(image_path)
    if(image is None):
        print('{} 画像を開けません。'.format(image_path))
        quit()

    # グレースケール変換
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(CASCADE_PATH + UPPER_BODY)
    # 物体認識処理
    # http://workpiles.com/2015/04/opencv-detectmultiscale-scalefactor/
    # http://workpiles.com/2015/04/opencv-detectmultiscale-minneighbors/
    rectangle = cascade.detectMultiScale(image_gray, scaleFactor=1.11, minNeighbors=4, minSize=(40, 40))

    if len(rectangle):
        # 顔だけ切り出し
        x = rectangle[0][0]
        y = rectangle[0][1]
        width = rectangle[0][2]
        height = rectangle[0][3]
        image = image[y:y+height, x:x+width]
        # 訓練用の画像サイズにリサイズ
        image = cv2.resize(image, (resize, resize))
        # 保存
        cv2.imwrite(new_image_path, image)
        print('write image. {}'.format(new_image_path))

    return new_image_path


def write_csv(outdir, csvname, data):
    writer = csv.writer(open(os.path.join(outDir, csvname), 'w'), lineterminator='\n')
    writer.writerows(data)


if __name__ == '__main__':
    random.seed()
    outDir = sys.argv[1]
    traincsv = 'train.csv'
    testcsv = 'test.csv'
    traindata = []
    testdata = []

    if not os.path.isdir(outDir):
        sys.exit('%s is not directory' % outDir)

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

    for dirpath, dirnames, filenames in os.walk(outDir):
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
                            image_path = pre_work(image_path)
                            if os.path.exists(image_path):
                                rand = random.random()
                                if rand > 0.2:
                                    traindata.append([image_path, labelnumber])
                                else:
                                    testdata.append([image_path, labelnumber])
                print("{} Done.".format(member_dir))

    write_csv(outDir, traincsv, traindata)
    write_csv(outDir, testcsv, testdata)
