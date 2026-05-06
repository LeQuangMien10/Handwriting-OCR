# ----------- Augmentation -----------

import albumentations as A

def get_train_transform():
    return A.Compose([
        A.GaussianBlur(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=3, p=0.3),
    ])

def get_val_transform():
    return A.Compose([])