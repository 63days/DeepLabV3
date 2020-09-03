import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import DataSetWrapper
from tqdm import tqdm
from unet import Unet, CenterCrop
from utils import get_IOU, Padding
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    valid_ratio = args.valid_ratio
    test_ratio = args.test_ratio
    threshold = args.threshold
    separable = args.separable
    method = args.method

    dataset = DataSetWrapper(batch_size, num_workers, valid_ratio, test_ratio)
    train_dl, valid_dl, test_dl = dataset.get_data_loaders()

    model = Unet(input_dim=1, separable=separable, method=method)
    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dl), eta_min=0,
                                                     last_epoch=-1)
    criterion = nn.BCEWithLogitsLoss()
    train_losses = []
    val_losses = []
    mIOU_list = []
    best_mIOU = 0.
    step = 0
    for epoch in range(epochs):
        pbar = tqdm(train_dl)
        model.train()
        losses = []
        for (img, label) in pbar:
            optimizer.zero_grad()
            img, label = img.to(device), label.to(device)
            pred = model(img)
            pred = Padding()(pred, label.size(3))
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            pbar.set_description(f'E: {epoch+1} | L: {loss.item():.4f} | lr: {scheduler.get_lr()[0]:.7f}')
        scheduler.step()
        if (epoch + 1) % 10:
            losses = sum(losses) / len(losses)
            train_losses.append(losses)

        with torch.no_grad():
            model.eval()
            mIOU = []
            losses = []
            pbar = tqdm(valid_dl)
            for (img, label) in pbar:
                img, label = img.to(device), label.to(device)
                pred = model(img)
                pred = Padding()(pred, label.size(3))
                loss = criterion(pred, label)

                mIOU.append(get_IOU(pred, label, threshold=threshold))
                losses.append(loss.item())

            mIOU = sum(mIOU) / len(mIOU)
            mIOU_list.append(mIOU)
            if (epoch + 1) % 10:
                losses = sum(losses) / len(losses)
                val_losses.append(losses)

            print(f'VL: {loss.item():.4f} | mIOU: {100 * mIOU:.1f}%')

        if mIOU > best_mIOU:
            best_mIOU = mIOU
            save_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_mIOU': best_mIOU
            }
            torch.save(save_state, f'./checkpoint/best_model.ckpt')
            step = 0
        else:
            step += 1
            if step > args.patience:
                print('Early stopped...')
                return


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
        default=4
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=200,
        help='Early Stopping Criteria'
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
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for pixel value of predicted image'
    )
    parser.add_argument(
        '--separable',
        action='store_true',
        default=False,
        help='Using Depth-Wise Separable Conv'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='upsample',
        choices=['upsample', 'transpose'],
        help='Upsample Method'
    )
    args = parser.parse_args()
    main(args)