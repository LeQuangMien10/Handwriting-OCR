import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import OCRDataset
from data.collate import collate_fn
from data.vocab import ALL_CHARS
from data.transforms import get_train_transform

from models.crnn import CRNN

import configs.config as cfg

def main():
    train_dataset = OCRDataset(
        root_dir="/kaggle/input/vietocr/InkData_line_processed",
        # transform=get_train_transform()
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = CRNN(num_classes=len(ALL_CHARS) + 1)

    # print(model)

    imgs, labels, label_lengths = next(iter(train_loader))

    print("Images:", imgs.shape)

    outputs = model(imgs)

    print("Model outputs:", outputs.shape)

    log_probs = outputs.log_softmax(2)

    input_lengths = torch.full(
        size=(imgs.size(0),),
        fill_value=outputs.size(0),
        dtype=torch.long,
    )

    print("Input lengths:", input_lengths)
    print("Label lengths:", label_lengths)

    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True
    )

    loss = criterion(
        log_probs,
        labels,
        input_lengths,
        label_lengths,
    )

    print("CTC Loss:", loss.item())

if __name__ == '__main__':
    main()
