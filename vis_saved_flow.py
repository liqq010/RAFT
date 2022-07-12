import sys
sys.path.append('core')
import os
import cv2
import numpy as np
import glob
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).float()
    return img[None].to(DEVICE)

path = sys.argv[1]
print(f'path: {path}')

images = glob.glob(os.path.join(path, '*.png'))
images = sorted(images)
print(images)


for img_file in images:
    print(img_file)
    img = load_image(img_file)[0]
    img = img.cpu().numpy()
    
    flo = img[:,:,0:2]
    print(flo.shape)
    flo = flow_viz.flow_to_image(flo)
    cv2.imshow('image',flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()
