import cv2
import matplotlib.pyplot as plot
import numpy as np
import os

# cascade path
CASCADE_PATH = "opencv_cascade_classifier/"
# cascade file
UPPER_BODY = CASCADE_PATH + "haarcascade_upperbody.xml"
FULL_BODY = CASCADE_PATH + "haarcascade_fullbody.xml"
LOWER_BODY = CASCADE_PATH + "haarcascade_lowerbody.xml"


def detectObjectFromImage(image_path, cascade_file):
    path = os.path.splitext(image_path)
    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(image_path)
    if image is None:
        print('{} 画像を開けません。'.format(image_path))
        quit()

    # グレースケール変換
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectangles = cascade.detectMultiScale(image_gray, scaleFactor=1.01, minNeighbors=1, minSize=(15, 15))

    count = 1
    for rectangle in rectangles:
        # オブジェクト切り出し
        x = rectangle[0]
        y = rectangle[1]
        width = rectangle[2]
        height = rectangle[3]
        rectangle_image = image[y:y + height, x:x + width]
        # 訓練用の画像サイズにリサイズ
        rectangle_image = cv2.resize(rectangle_image, (100, 100))
        plot.imshow(np.array(rectangle_image))

        new_image_path = path[0] + "_" + str(count) + path[1]
        cv2.imwrite(new_image_path, rectangle_image)
        count += 1


if __name__ == '__main__':
    detectObjectFromImage("images/superMacho/hiroe_ゴリマッチョ3.jpg", UPPER_BODY)
