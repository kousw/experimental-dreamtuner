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

class DreamboothDataset(Dataset):
    def __init__(self, data_path, tokenizer, regular_prompt, subject_prompt, placeholder_token, size=512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.regular_prompt = regular_prompt
        self.subject_prompt = subject_prompt.format(placeholder_token)
        print("subject_prompt", self.subject_prompt)
        self.placeholder_token = placeholder_token
        self.size = size
        # self.transform = transform
        self.regular_image_files = []
        self.subject_image_files = []
        # self.apply_default_transforms = apply_default_transforms
        
        self.clip_image_processor = CLIPImageProcessor()
        
        
        self.image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.subject_encoder_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: self.clip_image_processor(images=x, return_tensors="pt").pixel_values[0]),
        ])
        
        
        self.load_data()
        

    def load_data(self):
        # Load image, mask, and caption file paths
        
        for file in glob.glob(f'{self.data_path}/regular/*.png'):
            self.regular_image_files.append(file)
                         
        for file in glob.glob(f'{self.data_path}/subject/*.png'):
            self.subject_image_files.append(file)
        
        self.num_regular_images = len(self.regular_image_files)
        self.num_subject_images = len(self.subject_image_files)
        
        self.max_length = max(self.num_regular_images, self.num_subject_images)
        print(self.max_length, self.num_regular_images, self.num_subject_images)
    
    def __len__(self):
        return self.max_length

    def __getitem__(self, idx):
        
        example = {}
        regular_image = Image.open(self.regular_image_files[idx % self.num_regular_images])
        
        example["regular_images"] = self.image_transforms(regular_image)
        example["regular_images_for_se"] = self.subject_encoder_transforms(regular_image)
        
        # get text and tokenize        
        regular_prompt_inputs = self.tokenizer(
            self.regular_prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )          
        
        example["regular_prompt_ids"] = regular_prompt_inputs.input_ids
        example["regular_attention_mask"] = regular_prompt_inputs.attention_mask

        subject_image = Image.open(self.subject_image_files[idx % self.num_subject_images])
        
        example["subject_images"] = self.image_transforms(subject_image)
        example["subject_images_for_se"] = self.subject_encoder_transforms(subject_image)
        
        # get text and tokenize
        subject_prompt_inputs = self.tokenizer(
            self.subject_prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        example["subject_prompt_ids"] = subject_prompt_inputs.input_ids
        example["subject_attention_mask"] = subject_prompt_inputs.attention_mask
     
        return example
