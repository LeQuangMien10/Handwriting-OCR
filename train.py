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

    print(model)

    imgs, labels, lengths = next(iter(train_loader))

    outputs = model(imgs)

    print(outputs.shape)


if __name__ == '__main__':
    main()
