import argparse
import os
import random
import numpy as np
from torch import optim
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cls import Illumination_classifier


def test(model, test_loader):

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy


def init_seeds(seed=0):

    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Illumination Classifier Training')
    parser.add_argument('--dataset_path', default='datasets/cls_dataset', help='path to dataset')
    parser.add_argument('--save_path', default='pretrained', help='path to save pretrained models')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--image_size', default=64, type=int, help='input image size')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training')
    parser.add_argument('--cuda', default=True, type=bool, help='use GPU or not')

    args = parser.parse_args()


    init_seeds(args.seed)


    train_dataset = datasets.ImageFolder(
        args.dataset_path,
        transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])
    )

    image_nums = len(train_dataset)
    train_nums = int(image_nums * 0.9)
    test_nums = image_nums - train_nums
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_nums, test_nums])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    model = Illumination_classifier(input_channels=3)
    if args.cuda:
        model = model.cuda()


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_accuracy = 0.0


    for epoch in range(args.start_epoch, args.epochs):

        if epoch < args.epochs // 2:
            lr = args.lr
        else:
            lr = args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        train_tqdm = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}')
        for image, label in train_tqdm:
            if args.cuda:
                image, label = image.cuda(), label.cuda()

            optimizer.zero_grad()
            output = model(image)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()

            train_tqdm.set_postfix(loss=loss.item())


        accuracy = test(model, test_loader)


        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'{args.save_path}/best_cls.pth')

    print(f'Best Accuracy: {best_accuracy:.2f}%')