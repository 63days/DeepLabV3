import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import DataSetWrapper
from tqdm import tqdm
from unet import Unet, CenterCrop
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    valid_ratio = args.valid_ratio
    test_ratio = args.test_ratio

    dataset = DataSetWrapper(batch_size, num_workers, valid_ratio, test_ratio)
    train_dl, valid_dl, test_dl = dataset.get_data_loaders()

    model = Unet(1)
    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):

        pbar = tqdm(train_dl)
        model.train()
        for (img, label) in pbar:
            optimizer.zero_grad()
            img, label = img.to(device), label.to(device)
            pred = model(img)
            S = pred.size(2)
            label = CenterCrop()(label, S)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'E: {epoch+1} | L: {loss.item():.4f}')

        with torch.no_grad():
            model.eval()
            for (img, label) in valid_dl:
                img, label = img.to(device), label.to(device)
                pred = model(img)
                S = pred.size(2)
                label = CenterCrop()(label, S)
                loss = criterion(pred, label)
            print(f'VL: {loss.item():.4f}')

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
        default=8
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8
    )
    parser.add_argument(
        '--valid_ratio',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-4
    )
    args = parser.parse_args()
    main(args)