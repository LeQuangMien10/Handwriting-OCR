import os
import cv2
import torch
from torch.utils.data import Dataset

from data.vocab import char2idx

from configs.config import IMG_HEIGHT

class OCRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for file in os.listdir(root_dir):
            if file.endswith(".png"):
                img_path = os.path.join(root_dir, file)
                txt_path = os.path.join(root_dir, file.replace(".png", ".txt"))

                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    self.samples.append((img_path, text))

        # sort samples width-wise
        # self.samples.sort(key=lambda x: len(x[1]))

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        return [char2idx[c] for c in text if c in char2idx]

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        # load images
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize height = 32
        h, w, _ = img.shape
        new_h = IMG_HEIGHT
        new_w = int(w * (new_h / h))
        img = cv2.resize(img, (new_w, new_h))

        if self.transform:
            img = self.transform(image=img)["image"]

        # normalize
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        label = self.encode_text(text)

        return img, torch.tensor(label), len(label)