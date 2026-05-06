from torch.utils.data import DataLoader

from data.dataset import OCRDataset
from data.collate import collate_fn
from data.transforms import get_train_transform
import configs.config as cfg

def main():
    train_dataset = OCRDataset(
        root_dir="/kaggle/input/vietocr/InkData_line_processed",
        transform=get_train_transform()
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn
    )

    for imgs, labels, lengths in train_loader:
        print("Images:", imgs.shape)
        print("Labels:", labels.shape)
        print("Lengths:", lengths)
        break


if __name__ == '__main__':
    main()
