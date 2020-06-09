import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import utils
from model import Model, Omniglot_Model


class Net(nn.Module):
    def __init__(self, num_class, pretrained_path, resnet_depth=18):
        super(Net, self).__init__()

        if resnet_depth == 18:
            resnet_output_dim = 512
        elif resnet_depth == 34:
            resnet_output_dim = 512
        elif resnet_depth == 50:
            resnet_output_dim = 2048

        # encoder
        self.f = Model(resnet_depth=resnet_depth).f
        # classifier
        self.fc = nn.Linear(resnet_output_dim, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


class Omniglot_Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Omniglot_Net, self).__init__()

        # encoder
        self.f = Omniglot_Model().f
        # classifier
        self.fc = nn.Sequential(
            nn.Linear(1024, num_class, bias=False),
            #nn.BatchNorm1d(256),
            #nn.ReLU(inplace=True),
            #nn.Linear(256, 256, bias=True),
            #nn.BatchNorm1d(256),
            #nn.ReLU(inplace=True),
            #nn.Linear(256, num_class, bias=True)
        )
        
        
        
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        feature = self.f(x)
        feature = F.normalize(feature, dim=-1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of sweeps over the dataset to train')
    parser.add_argument('--resnet_depth', default=18, type=int, help='The depth of the resnet')
    parser.add_argument('--dataset', default='omniglot', type=str, help='omniglot or cifar')

    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    if args.dataset == 'cifar':
        resnet_depth = args.resnet_depth
        
        train_data = CIFAR10(root='data', train=True, transform=utils.train_transform, download=True)
        test_data = CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    else:
        train_data = utils.Our_Omniglot(root='data', background=False, transform=utils.omniglot_train_transform,
                                         character_target_transform=None, alphabet_target_transform=None, download=True,
                                         eval_split_train=True, out_character=False, contrast_training=False)
        test_data = utils.Our_Omniglot(root='data', background=False, transform=utils.omniglot_test_transform,
                                         character_target_transform=None, alphabet_target_transform=None, download=True,
                                         eval_split_train=False, out_character=False, contrast_training=False)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    
    if args.dataset == 'cifar':
        model = Net(num_class=len(train_data.classes), pretrained_path=model_path, resnet_depth=resnet_depth ).cuda()
    else:
        #model = Omniglot_Net(num_class=20, pretrained_path=model_path).cuda()
        model = Omniglot_Net(num_class=659, pretrained_path=model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False

    if args.dataset == 'cifar':
        flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    else:
        flops, params = profile(model, inputs=(torch.randn(1, 1, 28, 28).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    #optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    save_name_pre = model_path.split('.pth')[0]
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}_linear_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), '{}_linear_model.pth'.format(save_name_pre))
