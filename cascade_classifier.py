import cv2
import os
import sys
import CONST as C


def detectObjectFromImage(image_path, cascade_file):
    rect_image_paths = []
    image = cv2.imread(image_path)
    if image is None:
        print("can not read {}".format(image_path))
    elif C.PREFIX_EXTRACTED in image_path:
        print("already detect {}".format(image_path))
    else:
        # グレースケール変換
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_file)
        rectangles = cascade.detectMultiScale(image_gray, scaleFactor=1.01, minNeighbors=1, minSize=(30, 30))

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
            rect_image_path = path[0] + C.PREFIX_EXTRACTED + str(count) + path[1]
            cv2.imwrite(rect_image_path, rectangle_image)
            rect_image_paths.append(rect_image_path)
            count += 1

    return rect_image_paths


# 指定した画像(path)の物体を検出し、外接矩形の画像を出力します
def detect_contour(image_path):
    rect_image_paths = []
    # 画像を読込
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("can not read {}".format(image_path))
    elif C.PREFIX_RECTANGLED in image_path:
        print("already detect {}".format(image_path))
    else:
        paths = os.path.splitext(image_path)

        # グレースケール画像へ変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2値化
        retval, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 輪郭を抽出
        #   contours : [領域][Point No][0][x=0, y=1]
        #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
        #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
        _, contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 矩形検出された数（デフォルトで0を指定）
        detect_count = 0

        # 各輪郭に対する処理
        for i in range(0, len(contours)):

            # 輪郭の領域を計算
            area = cv2.contourArea(contours[i])

            # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
            if area < 1e2 or 1e5 < area:
                continue

            # 外接矩形
            if len(contours[i]) > 0:
                rect = contours[i]
                x, y, w, h = cv2.boundingRect(rect)
                # cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 外接矩形毎に画像を保存
                rect_image_path = paths[0] + C.PREFIX_RECTANGLED + str(detect_count) + paths[1]
                cv2.imwrite(rect_image_path, image[y:y + h, x:x + w])
                rect_image_paths.append(rect_image_path)

                detect_count = detect_count + 1

    return rect_image_paths


if __name__ == '__main__':
    print(detectObjectFromImage(sys.argv[1], C.UPPER_BODY))
