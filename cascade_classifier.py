import cv2
import os
import sys

# cascade path
CASCADE_PATH = "opencv_cascade_classifier/"
# cascade file
UPPER_BODY = CASCADE_PATH + "haarcascade_upperbody.xml"
FULL_BODY = CASCADE_PATH + "haarcascade_fullbody.xml"
LOWER_BODY = CASCADE_PATH + "haarcascade_lowerbody.xml"
# prefix
PREFIX_EXTRACTED = "_ext"


def detectObjectFromImage(image_path, cascade_file):
    rect_image_paths = []
    image = cv2.imread(image_path)
    if image is None:
        print("can not read {}".format(image_path))
    elif PREFIX_EXTRACTED in image_path:
        print("already extracted {}".format(image_path))
    else:
        image = cv2.resize(image, (100, 100))
        # グレースケール変換
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_file)
        rectangles = cascade.detectMultiScale(image_gray, scaleFactor=1.01, minNeighbors=1, minSize=(15, 15))

        path = os.path.splitext(image_path)
        count = 1
        for rectangle in rectangles:
            x = rectangle[0]
            y = rectangle[1]
            width = rectangle[2]
            height = rectangle[3]

            # オブジェクト切り出し
            rectangle_image = image[y:y + height, x:x + width]
            rectangle_image = cv2.resize(rectangle_image, (100, 100))

            # イメージ保存
            rect_image_path = path[0] + PREFIX_EXTRACTED + str(count) + path[1]
            cv2.imwrite(rect_image_path, rectangle_image)
            rect_image_paths.append(rect_image_path)
            count += 1

    return rect_image_paths


if __name__ == '__main__':
    print(detectObjectFromImage(sys.argv[1], UPPER_BODY))
