import argparse
import torch
from dataloader import DataSetWrapper
from tqdm import tqdm

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    valid_ratio = args.valid_ratio
    test_ratio = args.test_ratio

    dataset = DataSetWrapper(batch_size, num_workers, valid_ratio, test_ratio)
    train_dl, valid_dl, test_dl = dataset.get_data_loaders()

    for epoch in range(epochs):

        pbar = tqdm(train_dl)
        for (img, label) in pbar:
            pass

        with torch.no_grad():
            for (img, label) in valid_dl:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    parser.add_argument(
        '--epochs',
        type=int,
        default=int(1e5)
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8
    )
    parser.add_arguement(
        '--valid_ratio',
        type=float,
        default=0.2
    )
    parser.add_arguement(
        '--test_ratio',
        type=float,
        default=0.2
    )

    args = parser.parse_args()
    main(args)