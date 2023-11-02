import numpy as np
from PIL import Image
import os
import glob
from params import *

def load_and_process_image(img_file, img_sz):
    img = Image.open(img_file).resize(img_sz, Image.NEAREST)
    img = np.array(img) / 255.0
    img = (img - mu) / std
    return img

def load_and_process_label(label_file, img_sz):
    label = Image.open(label_file).convert("L").resize(img_sz, Image.NEAREST)
    label = np.array(label) / 255.0
    return label

if __name__ == "__main":
    trg_files = sorted(glob.glob(TRG_PATH + "*.png"), key=lambda img_file: int(os.path.basename(img_file).split('.')[0]))
    img_files = sorted(glob.glob(IMG_PATH + "*.jpg"), key=lambda img_file: int(os.path.basename(img_file).split('.')[0]))
    
    img_sz = (2800, 1760)
    
    imgs = [load_and_process_image(img_file, img_sz) for img_file in img_files]
    labels = [load_and_process_label(trg_file, img_sz) for trg_file in trg_files]
    
    np.save('imgs.npy', np.stack(imgs))
    np.save('labels.npy', np.stack(labels))
