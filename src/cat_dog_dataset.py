import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.dog_root = os.path.join(root_dir, "Dog")
        self.cat_root = os.path.join(root_dir, "Cat")

        self.filenames = []
        self.labels = []

        for dog_img in os.listdir(self.dog_root):
            self.filenames.append(os.path.join(self.dog_root, dog_img))
            self.labels.append(0)

        for cat_img in os.listdir(self.cat_root):
            self.filenames.append(os.path.join(self.cat_root, cat_img))
            self.labels.append(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label
