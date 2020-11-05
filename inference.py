# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger, get_model_summary

import dataset
import models
from core.inference import get_final_preds, get_final_preds_using_softargmax
from utils.transforms import get_affine_transform
from models.lpn import get_pose_net
from utils.transforms import flip_back
import cv2
import numpy as np
import time
from yolact.inference import InferenceEngine

crop_fraction = 0.15
input_size = 512

wight = 192
hight = 256
aspect_ratio = wight * 1.0 / hight

point_color2 = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
                (240, 2, 127), (240, 2, 127),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (252, 176, 243), (0, 176, 240), (252, 176, 243),
                (0, 176, 240), (252, 176, 243), (0, 176, 240),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142)]


def _xywh2cs( x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / 200, h * 1.0 / 200],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale



def get_yolactjit_result():

    img1_ori = cv2.imread("/root/yolact_online/0001.png")
    img1_crop = img1_ori[:, int(crop_fraction * img1_ori.shape[0]):-int(crop_fraction * img1_ori.shape[0])]
    img1 = cv2.cvtColor(cv2.resize(img1_crop, (input_size, input_size)), cv2.COLOR_BGR2RGB)
    img2_ori = cv2.imread("/root/yolact_online/0056.png")
    img2_crop = img2_ori[:, int(crop_fraction * img2_ori.shape[0]):-int(crop_fraction * img2_ori.shape[0])]
    img2 = cv2.cvtColor(cv2.resize(img2_crop, (input_size, input_size)), cv2.COLOR_BGR2RGB)
    img3_ori = cv2.imread("/root/yolact_online/0023.png")
    img3_crop = img3_ori[:, int(crop_fraction * img3_ori.shape[0]):-int(crop_fraction * img3_ori.shape[0])]
    img3 = cv2.cvtColor(cv2.resize(img3_crop, (input_size, input_size)), cv2.COLOR_BGR2RGB)
    img4_ori = cv2.imread("/root/yolact_online/0011.png")
    img4_crop = img4_ori[:, int(crop_fraction * img4_ori.shape[0]):-int(crop_fraction * img4_ori.shape[0])]
    img4 = cv2.cvtColor(cv2.resize(img4_crop, (input_size, input_size)), cv2.COLOR_BGR2RGB)

    engine = InferenceEngine("/root/yolact_online/weights/yolact_plus_im512_VoVonet_19_94_149625.pth",
                             "yolact_plus_im512_VoVonet_19_config")
    engine.input_images = [img1, img2, img3, img4]
    engine.infer()
    preds = engine.get_preds()

    result = []
    for index, img in enumerate([img1_crop, img2_crop, img3_crop, img4_crop]):
        output = {}

        temp_img = []
        c_list = []
        s_list = []
        for box_index, box_score in enumerate(preds[index]['score']):
            if box_score > 0.6:
                temp_box = preds[index]['box']
                temp_box[box_index, 0] *= img.shape[1]
                temp_box[box_index, 1] *= img.shape[0]
                temp_box[box_index, 2] *= img.shape[1]
                temp_box[box_index, 3] *= img.shape[0]
                y1 = temp_box[box_index, 1]
                x1 = temp_box[box_index, 0]
                y2 = temp_box[box_index, 3]
                x2 = temp_box[box_index, 2]

                c, s = _xywh2cs(x1, y1, x2 - x1, y2 - y1)

                trans = get_affine_transform(c, s, 0, np.array([wight, hight]))
                input = cv2.warpAffine(
                    img,
                    trans,
                    (int(wight), int(hight)),
                    flags=cv2.INTER_LINEAR)

                # img_crop = img[ int(y1):int(y2), int(x1):int(x2), :]
                # img_crop = cv2.resize(img_crop, (wight,hight))
                temp_img.append(input)

                c_list.append(c)
                s_list.append(s)
        output['full_image'] = img
        output['crop_image'] = temp_img
        output['c'] = c_list
        output['s'] = s_list

        result.append(output)



    return result





def main():


    yolact_result = get_yolactjit_result()
    args = argparse.Namespace()
    args.cfg = 'experiments/coco/lpn/lpn50_256x192_gd256x2_gc.yaml'
    args.modelDir = ''
    args.logDir = ''
    update_config(cfg, args)
    model = get_pose_net(cfg, is_train=False)
    #model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)

    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    model = model.cuda()
    #model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    Fasttransforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    model.eval()

    for image_index, it in enumerate(yolact_result):

        transforms_image = []
        for crop_image in it['crop_image']:
            crop_image = Fasttransforms(crop_image)
            transforms_image.append(crop_image)

        transforms_image = torch.from_numpy(np.stack(transforms_image)).cuda().half()
        #temp = np.stack(it['crop_image']).transpose(0, 3, 1, 2)

        #transforms_image = torch.cat((transforms_image, transforms_image, transforms_image, transforms_image, transforms_image), 0)[:50]
        outputs = model(transforms_image)

        torch.cuda.synchronize()
        t = time.time()
        outputs = model(transforms_image)
        torch.cuda.synchronize()
        print(time.time() - t)

        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        if cfg.TEST.FLIP_TEST:
            input_flipped = transforms_image.flip(3)
            outputs_flipped = model(input_flipped)

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped
            flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
            output_flipped = flip_back(output_flipped.cpu().detach().numpy(), flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if cfg.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.clone()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        preds, maxvals, preds_ori  = get_final_preds_using_softargmax(cfg, output.clone(), np.array(it['c']).copy(), np.array(it['s']).copy())
        temp_image = it['full_image'].copy()
        for point_for_image in(preds):
            for index, point in enumerate(point_for_image):
                cv2.circle(temp_image, (int(point[0]), int(point[1])),1, point_color2[index], 3)
        cv2.imwrite('result_{}.png'.format(image_index), temp_image)
        print("")

        print("")



if __name__ == '__main__':


    main()
