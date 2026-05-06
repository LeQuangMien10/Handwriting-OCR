# ----------- Collate function -----------
# Images have different widths, labels have different lengths → padding

import torch

def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)

    max_w = max(img.shape[2] for img in images)

    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad = torch.zeros((c, h, max_w))
        pad[:, :, :w] = img
        padded_images.append(pad)

    images = torch.stack(padded_images)

    labels = torch.cat(labels)
    label_lengths = torch.tensor(label_lengths)

    return images, labels, label_lengths