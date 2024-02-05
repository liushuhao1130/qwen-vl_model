import concurrent.futures
import json
import os
import random
import threading

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor

from utils import read_jsonl, scan_files_in_dir, scaling

IMAGE_SUFFIX = ['.jpg', '.png', '.jpeg']
class YOOX(Dataset):
    def __init__(self, meta_root, image_root, tokenizer=None, shuffle_caption=True, drop_caption: float = 0.05,
                 t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05,
                 **kwargs):
        meta_root = meta_root if meta_root is not None else "/home/chongzheng_p23/data/Datasets/YOOX/YOOX-Meta"
        self.shuffle_caption = shuffle_caption
        self.drop_caption = drop_caption  # TODO: drop_caption
        self.tokenizer = tokenizer
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.data = []
        self.captions = {}
        self.clip_image_processor = CLIPImageProcessor()
        self.error_num = 0
        if meta_root is not None and image_root is not None:
            self._scan_items(meta_root, image_root)

        print(f"Found {len(self.data)} items in {image_root}")

        self.transform = transforms.Compose([
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    def get_data(self):
        return self.data, self.captions
    def path_prepare(self, num_samples=2000, num_text=5, num_images=2):
        path_list = random.sample(self.data, num_samples)
        with open("path.jsonl", 'w') as file:
            for path in path_list:
                same_style_image = []
                while len(same_style_image) != num_images:
                    item_id = os.path.basename(path).split("-")[0]
                    caption = self.captions[item_id].split("<&>")
                    select_words = random.sample(caption,num_text)
                    same_style_image = self.get_same_images(num_images, select_words)

                data = {
                    "image_path": path,
                    "prompt": ",".join(select_words),
                    "style_list": same_style_image
                }
                json_line = json.dumps(data)
                file.write(json_line + "\n")
    def _scan_items(self, meta_root, image_root):
        progress = tqdm(desc="Dataset Scanning", total=0)
        for gender in os.scandir(image_root):
            if gender.is_dir():
                for cate in os.scandir(gender.path):
                    if cate.is_dir():
                        for gory in os.scandir(cate.path):
                            item_jsonl = os.path.join(gory.path, "items.jsonl").replace(image_root, meta_root)
                            if not os.path.exists(item_jsonl):
                                continue
                            item_caption = self._read_yoox_caption(item_jsonl)
                            self.captions.update(item_caption)

                            items, add_num = os.scandir(gory.path), 0
                            for item in items:
                                if item.is_dir() and item.name in item_caption:
                                    self.data.extend(item_images := [_.path for _ in os.scandir(item.path)
                                                                     if not _.name.endswith("-1.jpg")
                                                                     and not _.name.endswith("-2.jpg")
                                                                     and not _.name.endswith("-3.jpg")])
                                    add_num += len(item_images)

                            progress.total += add_num
                            progress.update(add_num)

    @staticmethod
    def _read_yoox_caption(item_jsonl):
        meta_data = read_jsonl(item_jsonl)
        captions = {}
        for item in meta_data:
            item_id = item.pop("id")
            try:
                item.pop("image_urls")
            except KeyError:
                print(item_id, item)
            item_captions = set()

            def iter_dict(info):
                for key, value in info.items():
                    if isinstance(value, list):
                        for v in value:
                            yield v.replace(", ", "<&>")
                    elif isinstance(value, dict):
                        yield from iter_dict(value)
                    elif isinstance(value, str):
                        yield value.replace(", ", "<&>")

            item_captions.update(iter_dict(item))
            captions[item_id] = "<&>".join(item_captions)
        return captions
    @staticmethod
    def check_image(image_path):
            try:
                image = Image.open(image_path)
            except:
                return False
            if len(image.mode) != 3:
                return False

            channels = image.split()
            channel_sizes = [channel.size for channel in channels]
            if not all(size == channel_sizes[0] for size in channel_sizes):
                return False

            return True

    def __len__(self):
        return len(self.data)
    def get_random_images(self, image_path, num_images):

        folder_path = os.path.dirname(image_path)
        # 获取文件夹下所有图片文件
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # 随机选择指定数量的图片文件
        selected_images = random.sample(image_files, min(num_images, len(image_files)))

        # 加载并返回选定的图片
        images = []
        for image_file in selected_images:
            image_path = os.path.join(folder_path, image_file)
            if self.check_image(image_path):
                 img = Image.open(image_path)
                 images.append(img)
        return images
    def get_same_images(self, num_images, select_words, max_attempt=1000):
        image_list = []
        caption_list = []
        for i in range(max_attempt):
            rand_index = np.random.randint(0, self.__len__())
            image_path = self.data[rand_index]
            item_id = os.path.basename(image_path).split("-")[0]
            caption = self.captions[item_id].split("<&>")
            set_caption = set(caption)
            if set_caption.issuperset(set(select_words)):
                # image = Image.open(image_path)
                image_list.append(image_path)
                caption_list.append(caption)
                if len(image_list) == num_images:
                    break

        return image_list, caption_list
    def __getitem__(self, idx):
        image_path = self.data[idx]
        # raw_image = Image.open(image_path)
        # self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        num_images = 2
        num_text = 5
        num_round = 5
        item_id = os.path.basename(image_path).split("-")[0]
        caption = self.captions[item_id].split("<&>")
        image_list = []
        for i in range(num_round):
            if len(caption) > num_text:
                select_words = random.sample(caption, num_text)
            else:
                select_words = caption
            list, _ = self.get_same_images(num_images, select_words)
            if len(list) > len(image_list):
                image_list = list
            if len(image_list) == num_images:
                break

        caption = ", ".join(caption)
        return {
            "image_path": image_path,
            "select_words": select_words,
            "caption": caption,
            "image_list": image_list
        }




