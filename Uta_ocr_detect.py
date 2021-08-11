"""
coding: utf-8
python: 3.7.6
Date: 2021-03-05 11:55:48
@Software: VsCode
@Author: Gaopeng Bai
@Email: gaopengbai0121@gmail.com
@Description: 
"""
from numpy import random
from pathlib import Path

import cv2
import torch

from utils.general import (
    check_img_size,
    increment_path,
    non_max_suppression,
    scale_coords,
    set_logging,
    xyxy2xywh,
)
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.datasets import letterbox
import numpy as np


def Uta_Ocr_detect(
    weights="runs/train/uta100s/weights/best.pt",
    path="data/images/A1001001A18A10B22054H03266_0.png",
):
    imgsz = 640

    # Initialize
    set_logging()
    device = select_device("cpu")
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once

    img0 = cv2.imread(path)  # BGR
    assert img0 is not None, "Image Not Found " + path

    # Padded resize
    img = letterbox(img0, 640, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Recognition
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)

    # Process detections
    det = pred[0]
    label, position, averconf = [], [], []

    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):

            # for digital ocr recognition
            try:
                labelint = int(f"{names[int(cls)]}")
            except:
                labelint = 10
            xywh = (
                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            )  # normalized xywh
            position.append(xywh[0])
            averconf.append(conf)
            label.append(labelint)

    rank = [
        index for index, value in sorted(list(enumerate(position)), key=lambda x: x[1])
    ]
    sorted_position = [position[i] for i in rank]
    sorted_labels = [label[i] for i in rank]

    for i, value in enumerate(sorted_position):
        if i == len(sorted_position) - 1:
            break
        diff = sorted_position[i + 1] - value
        if diff < 0.01:
            sorted_position.pop(i + 1)
            sorted_labels.pop(i + 1)

    return {
        "text": "".join("." if e == 10 else str(e) for e in sorted_labels),
        "conf": np.mean(averconf),
    }


if __name__ == "__main__":

    with torch.no_grad():
        print(
            Uta_Ocr_detect(
                weights="runs/train/uta500l/weights/best.pt",
                path="data/images/A1001001A18A09A94009A00966_0.png",
            )
        )
