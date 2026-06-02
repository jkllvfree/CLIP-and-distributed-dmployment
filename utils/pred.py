import cv2
import re
import urllib.request
from typing import List
import numpy as np
import torch
from torchvision import transforms
from model import clip_loader as m

state_dict = torch.jit.load('ViT-L-14.pt', map_location='cpu').state_dict()
transform = transforms.ToTensor()


def read_image_from_path(path: str):
    """自动判断 path 是 URL 还是本地路径，读取并返回 BGR numpy array"""
    if re.match(r'^https?://', path):
        resp = urllib.request.urlopen(path)
        image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path)

    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def predict(model, image_paths: List[str], class_names: List[str]):
    prompts = ['a photo of a ' + class_name for class_name in class_names]
    tokenized_prompts = m.tokenize(prompts).to(torch.device('cuda'))

    imgs = []
    for path in image_paths:
        img = read_image_from_path(path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img)  # [C, H, W]
        imgs.append(img)

    imgs = torch.stack(imgs).to(torch.device('cuda'))  # [B, C, H, W]


    with torch.no_grad():
        logits_per_image, logits_per_text = model(imgs, tokenized_prompts)

    return logits_per_image, logits_per_text