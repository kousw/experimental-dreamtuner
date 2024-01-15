import json
import os
import io
import csv
import math
import random
import numpy as np
from einops import rearrange
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import glob
from transformers import CLIPImageProcessor

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, enable_mask=False, i_drop_rate=0.05, t_drop_rate=0.05, ti_drop_rate=0.05, apply_default_transforms=True, transform=None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.enable_mask = enable_mask
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.transform = transform
        self.ids = []
        self.image_files = []
        self.image_wo_bg_files = []
        self.depth_files = []
        self.mask_files = []
        self.image_face_files = []
        self.image_face_wo_bg_files = []
        self.face_depth_files = []
        self.face_mask_files = []
        self.tag_files = []
        self.apply_default_transforms = apply_default_transforms
        self.use_face_prob = 0.3
        
        self.clip_image_processor = CLIPImageProcessor()
        
        self.load_data()
        
        self.image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.image_wo_bg_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: self.clip_image_processor(images=x, return_tensors="pt").pixel_values[0]),
        ])
        
        self.depth_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ])
        
        self.mask_transforms = transforms.Compose(
        [                       
            transforms.ToTensor(),
        ])

    def load_data(self):
        # Load image, mask, and caption file paths
        
        files = []
        # sort files by name
        
        for file in glob.glob(f'{self.data_path}/*[0-9].png'):
            file = os.path.basename(file)
            file_without_ext = file.split('.')[0]
            files.append(file_without_ext)
                         
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                
        for file in files:                
            self.ids.append(file)
            self.image_files.append(os.path.join(self.data_path, file + '.png'))            
            self.image_wo_bg_files.append(os.path.join(self.data_path, file + '_rgb.png'))
            self.depth_files.append(os.path.join(self.data_path, file + '_depth.png'))
            self.image_face_files.append(os.path.join(self.data_path, file + '_face.png'))
            self.image_face_wo_bg_files.append(os.path.join(self.data_path, file + '_face_rgb.png'))
            self.face_depth_files.append(os.path.join(self.data_path, file + '_face_depth.png'))
            self.tag_files.append(os.path.join(self.data_path, file + '_tags.json'))
            if self.enable_mask:
                self.mask_files.append(os.path.join(self.data_path, file + '_mask.png'))
                self.face_mask_files.append(os.path.join(self.data_path, file + '_face_mask.png'))
        
        
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        
        id = self.ids[idx]
        
        use_face = False
        
        # if you want to use face images, uncomment below
        # this makes the model to use face images with 30% probability to stabilize training
        # please make sure that you have face images in your dataset(*_face.png, *_face_rgb.png, *_face_depth.png, *_face_mask.png)        
        # if random.random() < self.use_face_prob:
        #    use_face = True
                   
        
        # Read image and mask
        if use_face:
            image = Image.open(self.image_face_files[idx])
            image_wo_bg = Image.open(self.image_face_wo_bg_files[idx])
            depth = Image.open(self.face_depth_files[idx])
            if self.enable_mask:
                mask = Image.open(self.face_mask_files[idx])
        else:
            image = Image.open(self.image_files[idx])        
            image_wo_bg = Image.open(self.image_wo_bg_files[idx])
            depth = Image.open(self.depth_files[idx])
            if self.enable_mask:    
                mask = Image.open(self.mask_files[idx])
        
        if image.mode != 'RGB':
            print("image mode is not RGB", image.mode)
            image = image.convert('RGB')
        
        if image_wo_bg.mode != 'RGB':
            print("image_wo_bg mode is not RGB", image_wo_bg.mode)
            image_wo_bg = image_wo_bg.convert('RGB')
        
        depth = depth.convert('RGB')
            
        # open caption file
        with open(self.tag_files[idx], 'r') as f:
            tags = f.read()       
            
        # apply random crop if not face
        if not use_face:
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(512, 512))
            image = transforms.functional.crop(image, i, j, h, w)
            image_wo_bg = transforms.functional.crop(image_wo_bg, i, j, h, w)
            depth = transforms.functional.crop(depth, i, j, h, w)
            if self.enable_mask:
                mask = transforms.functional.crop(mask, i, j, h, w)
        
        # get text
        tags = json.loads(tags)
        text = tags["general"]
            
        # drop        
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = "solo"
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = "solo"
            drop_image_embed = 1            
        
        # get text and tokenize        
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids          

        # Apply random rotation, flip, and scale
        if random.random() < 0.1:
            angle = random.randint(-90, 90)
            image = image.rotate(angle)
            image_wo_bg = image_wo_bg.rotate(angle)
            depth = depth.rotate(angle)
            if self.enable_mask:
                mask = mask.rotate(angle)

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_wo_bg = image_wo_bg.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            if self.enable_mask:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # if random.random() < 0.1:
        #     scale_factor = random.uniform(0.8, 1.2)
        #     new_size = tuple(int(dim * scale_factor) for dim in image.size)
        #     image = image.resize(new_size, Image.BILINEAR)
        #     image_wo_bg = image_wo_bg.resize(new_size, Image.BILINEAR)
        #     depth = depth.resize(new_size, Image.BILINEAR)
        #     mask = mask.resize(new_size, Image.BILINEAR)
        
        if self.apply_default_transforms:
            image = self.image_transforms(image)
            image_wo_bg = self.image_wo_bg_transforms(image_wo_bg)
            depth = self.depth_transforms(depth)
            if self.enable_mask:
                mask = self.mask_transforms(mask)
    

        # Apply additional transformations if specified
        if self.transform is not None:
            image = self.transform(image)
            image_wo_bg = self.transform(image_wo_bg)
            depth = self.transform(depth)
            if self.enable_mask:
                mask = self.transform(mask)
                
        # print ("image", image.shape, min(image.flatten()), max(image.flatten()))
        # print ("image_wo_bg", image_wo_bg.shape, min(image_wo_bg.flatten()), max(image_wo_bg.flatten()))
        # print ("depth", depth.shape, min(depth.flatten()), max(depth.flatten()))
        # if self.enable_mask:
        #     print ("mask", mask.shape, min(mask.flatten()), max(mask.flatten()))
        
        data = {
            "id" : id,
            "pixel_values" : image,
            "reference_pixel_values" : image_wo_bg,
            "conditioning_pixel_values" : depth,            
            "text_input_ids" : text_input_ids,
            "drop_image_embed" : drop_image_embed
        }
        
        if self.enable_mask:
            data["mask"] = mask
        
        return data


# test code as main
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image
    from torch.utils.data import DataLoader
    from torchvision import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/large', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    opt = parser.parse_args()

    dataset = CustomDataset(opt.data_path, apply_default_transforms=False, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    
    # show images and captions from dataloader
    
    for step, batch in enumerate(dataloader):
        # print(batch)
        images = batch["pixel_values"]
        image_wo_bgs = batch["reference_pixel_values"]
        depths = batch["conditioning_pixel_values"]
        # depths = depths.repeat(1, 3, 1, 1)
        
        grid = make_grid(torch.cat([images,image_wo_bgs, depths], 0), nrow=opt.batch_size)
        grid = grid.permute(1, 2, 0)  # チャンネルの順序を変更
        # plt.imshow(grid.numpy())  # numpy配列に変換して表示
        # plt.show()

        #for tag in tags:
        #    print(tag)