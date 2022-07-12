import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    print(flo.shape)

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    print(img.shape)
    print(flo.shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

def make_dir(path):
    if os.path.exists(path):
        return
    os.makedirs(path)

def save_flow(flow_save_path, vis_save_path, img_path, img, flo):
    folder = img_path.split('/')[-2]
    img_name = img_path.split('/')[-1]
    make_dir(os.path.join(flow_save_path, folder))
    
    flow_file = os.path.join(flow_save_path, folder, img_name)
    flow_file = flow_file.replace('jpg', 'png')
    
    flo = flo[0].permute(1,2,0).cpu().numpy()
    h = flo.shape[0]
    w = flo.shape[1]
    # np.save(flow_file, flo)
    flow = np.concatenate((flo, np.zeros((h,w,1))), axis=2)

    # save x, y flows to r and g channels, since opencv reverses the colors
    print(f'flow: {flow}')

    cv2.imwrite(flow_file, flow[:,:,::-1])
    exit()
        
    make_dir(os.path.join(vis_save_path, folder))
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=0)
    vis_flow_file = os.path.join(vis_save_path, folder, img_name)
    cv2.imwrite(vis_flow_file, flo[:, :, [2,1,0]])

    vis_tmp = '/home/liqq/project/699/RAFT/road_optical_flow_vis_tmp'
    make_dir(os.path.join(vis_tmp, folder))
    img_flo = np.concatenate([img, flo], axis=0)
    vis_flow_file_tmp = os.path.join(vis_tmp, folder, img_name)
    cv2.imwrite(vis_flow_file_tmp, img_flo[:, :, [2,1,0]])

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        print(images)
        
        for idx, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            if idx % 1000 == 0:
                print(f'{idx} / {len(images)}')
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            print(flow_up)
            # exit()
            # viz(image1, flow_up)
            save_flow(args.save_path, args.vis_save_path, imfile1, image1, flow_up)
            # exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--save_path', default='/home/liqq/project/699/RAFT/tmp', help='path to save optical flow')
    parser.add_argument('--vis_save_path', default='/home/liqq/project/699/RAFT/road_optical_flow_vis', help='path to save optical flow')
    args = parser.parse_args()

    # p = '../road-dataset/road/rgb-images/'
    # paths = [
    #     '2014-08-08-13-15-11_stereo_centre_01',
    #     '2014-08-11-10-59-18_stereo_centre_02',
    #     '2014-11-14-16-34-33_stereo_centre_06',
    #     '2014-11-18-13-20-12_stereo_centre_05',
    #     '2014-11-21-16-07-03_stereo_centre_01',
    #     '2014-11-25-09-18-32_stereo_centre_04',
    #     '2014-12-09-13-21-02_stereo_centre_01'
    # ]
    # paths = [
    #     '2015-02-03-08-45-10_stereo_centre_02',
    #     '2015-02-03-19-43-11_stereo_centre_04',
    #     '2015-02-06-13-57-16_stereo_centre_02',
    #     '2015-02-13-09-16-26_stereo_centre_02',
    #     '2015-02-13-09-16-26_stereo_centre_05',
    #     '2015-02-24-12-32-19_stereo_centre_04',
    #     '2015-03-03-11-31-36_stereo_centre_01'
    # ]
    # for path in paths:
    #     args.path = p + path
    #     print(args.path)
    #     demo(args)
    demo(args)