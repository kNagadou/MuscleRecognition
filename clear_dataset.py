import os
import glob
import CONST as C

if __name__ == '__main__':
    [os.remove(f) for f in glob.glob(os.path.join(C.DATASET_DIR, '*', '*' + C.PREFIX_EXTRACTED + '*'))]
    [os.remove(f) for f in glob.glob(os.path.join(C.DATASET_DIR, '*', '*' + C.PREFIX_RECTANGLED + '*'))]
    [os.remove(f) for f in glob.glob(os.path.join(C.DATASET_DIR, '*', '*' + C.PREFIX_RESIZE + '*'))]
