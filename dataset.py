from os import path, walk
import torch
from torchvision import transforms
import random

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):

        if not path.exists(data_path):
            raise Exception(data_path + ' does not exist!')

        self.data = []

        for root, dirs, files in walk(data_path):
            for file in files:
                self.data.append(path.join(root, file))

        num_images = len(self.data)
        print(num_images)
        self.data = random.sample(self.data, num_images)  # only use num_images images

        # We use the transforms described in official PyTorch ResNet inference example:
        # https://pytorch.org/hub/pytorch_vision_resnet/.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]

        image = Image.open(image_path)

        try:
            image = self.transform(image)  # some images in the dataset cannot be processed - we'll skip them
        except Exception:
            return None

        dict_data = {
            'image': image,
            'image_path': image_path
        }
        return dict_data


# Skips empty samples in a batch
def collate_skip_empty(batch):
    batch = [sample for sample in batch if sample]  # check that sample is not None
    return torch.utils.data.dataloader.default_collate(batch)
