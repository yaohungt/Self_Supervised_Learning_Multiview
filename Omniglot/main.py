import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import math

import utils
from model import Model, Omniglot_Model, Recon_Omniglot_Model

from compute_MI_CondEntro import init_models, information


def contrastive_loss(out_1, out_2, _type='NCE'):
    # compute loss
    if _type == 'NCE':
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    elif _type == 'JS':
        temperature_JS = temperature
        
        scores_12 = torch.mm(out_1, out_2.t().contiguous()) / temperature_JS
        first_term = -F.softplus(-scores_12.diag()).mean()
        
        n = scores_12.size(0)
        second_term_12 = (torch.sum(F.softplus(scores_12)) -
                       torch.sum(F.softplus(scores_12.diag()))) / (n * (n - 1.))
        scores_11 = torch.mm(out_1, out_1.t().contiguous()) / temperature_JS
        second_term_11 = (torch.sum(F.softplus(scores_11)) -
                       torch.sum(F.softplus(scores_11.diag()))) / (n * (n - 1.))
        scores_22 = torch.mm(out_2, out_2.t().contiguous()) / temperature_JS
        second_term_22 = (torch.sum(F.softplus(scores_22)) -
                       torch.sum(F.softplus(scores_22.diag()))) / (n * (n - 1.))
        second_term = (second_term_11 + second_term_22 + second_term_12*2.) / 4.
        loss = -1. * (first_term - second_term)
    return loss


def inverse_perdictive_loss(feature_1, feature_2):
    # symmetric
    return F.mse_loss(feature_1, feature_2) + F.mse_loss(feature_2, feature_1)


# Use MSE_Loss here (assuming Gaussian)
# Other losses can be binary cross_entropy is assuming Bernoulli
def forward_predictive_loss(target, recon, _type='RevBCE'):
    if _type == 'RevBCE':
        # empirically good
        return F.binary_cross_entropy_with_logits(target, recon)
    elif _type == 'BCE':
        # assuming factorized bernoulli
        return F.binary_cross_entropy_with_logits(recon, target)
    elif _type == 'MSE':
        # assuming diagnonal Gaussian
        # target has [0,1]
        # change it to [-\infty, \infty]
        # inverse of sigmoid: x = ln(y/(1-y)) 
        target = target.clamp(min=1e-4, max=1. - 1e-4)
        target = torch.log(target / (1.-target))
        return F.mse_loss(recon, target)


# train for one epoch to learn unique features
def train(epoch, net, data_loader, train_optimizer, recon_net, loss_type, recon_optimizer, info_dct=None):
    net.train()
    if recon_net is not None:
        recon_net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2, target = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True),\
                               target.cuda(non_blocking=True)
        #norm = False if loss_type == 7 else True
        norm = True
        
        feature_1, out_1 = net(pos_1, norm)
        feature_2, out_2 = net(pos_2, norm)
        
        if loss_type == 1 or loss_type == 2 or loss_type == 5 or loss_type == 6 or loss_type == 7\
           or loss_type == 11 or loss_type == 12:
            contr_type = 'NCE' if not loss_type == 7 else 'JS'
            contr_loss = contrastive_loss(out_1, out_2, _type = contr_type)
        if loss_type == 2 or loss_type == 4 or loss_type == 6 or loss_type == 10\
           or loss_type == 12:
            inver_loss = inverse_perdictive_loss(feature_1, feature_2)
        if loss_type == 3 or loss_type == 4 or loss_type == 5 or loss_type == 6 or loss_type == 8\
           or loss_type == 9 or loss_type == 10 or loss_type == 11 or loss_type == 12:
            recon_for_2 = recon_net(feature_1)
            recon_for_1 = recon_net(feature_2)
            if loss_type == 3:
                recon_type = 'BCE'
            elif loss_type == 4 or loss_type == 5 or loss_type == 6 or loss_type == 8:
                recon_type = 'RevBCE'
            elif loss_type == 9 or loss_type == 10 or loss_type == 11 or loss_type == 12:
                recon_type = 'MSE'
            recon_loss = forward_predictive_loss(pos_1, recon_for_1, _type=recon_type) +\
                         forward_predictive_loss(pos_2, recon_for_2, _type=recon_type)
            recon_optimizer.zero_grad()
        train_optimizer.zero_grad()
            
        if loss_type == 1 or loss_type == 7:
            loss = contr_loss
        elif loss_type == 2:
            loss = contr_loss + inver_param*inver_loss
        elif loss_type == 3 or loss_type == 8 or loss_type == 9:
            loss = recon_param*recon_loss
        elif loss_type == 4 or loss_type == 10:
            loss = recon_param*recon_loss + inver_param*inver_loss
        elif loss_type == 5 or loss_type == 11:
            loss = contr_loss + recon_param*recon_loss
        elif loss_type == 6 or loss_type == 12:
            loss = contr_loss + recon_param*recon_loss +\
                   inver_param*inver_loss
        
        
        loss.backward()
        
        train_optimizer.step()
        if recon_optimizer is not None:
            recon_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}], loss_type: {}, Loss: {:.4f}'.format(\
                                   epoch, epochs, loss_type, total_loss / total_num))
        
        
        
    if info_dct is not None:
        inner_epochs = 200 if epoch < 100 else 80
        I_Z_S, I_Z_T, H_Z_T, H_Z_S = information(epoch, data_loader, inner_epochs, net,\
                info_dct['info_models'], info_dct['info_optimizer'])
        
        info_dct['info_results']['I(Z;S)'].append(I_Z_S)
        info_dct['info_results']['I(Z;T)'].append(I_Z_T)
        info_dct['info_results']['H(Z|T)'].append(H_Z_T)
        info_dct['info_results']['H(Z|S)'].append(H_Z_S)
        
    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def omniglot_test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Omniglot Experiments')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax\
                         (0.1 for omniglot and 0.5 for cifar)')
    parser.add_argument('--k', default=1, type=int, help='Top k most similar images used to predict the label\
                         (1 for omniglot and 200 for cifar)')
    parser.add_argument('--batch_size', default=482, type=int, help='Number of images in each mini-batch\
                         (964/2=482 for omniglot and 512 for cifar)')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--resnet_depth', default=18, type=int, help='The depth of the resnet\
                         (only for cifar)')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector\
                         (only for cifar)')
    parser.add_argument('--dataset', default='omniglot', type=str, help='omniglot or cifar')
    parser.add_argument('--trial', default=99, type=int, help='number of trial')
    parser.add_argument('--loss_type', default=1, type=int, help='1: only contrast (NCE),\
                        2: contrast (NCE) + inverse_pred, 3: only forward_pred (BCE),\
                        4: forward_pred (RevBCE) + inverse_pred, 5: contrast (NCE) + forward_pred (RevBCE),\
                        6: contrast (NCE) + forward_pred (RevBCE) + inverse_pred,\
                        7: only contrast (JS), 8: only forward_pred (RevBCE),\
                        9: only forward_pred (MSE), 10: forward_pred (MSE) + inverse_pred,\
                        11: contrast (NCE) + forward_pred (MSE),\
                        12: contrast (NCE) + forward_pred (MSE) + inverse_pred')
    parser.add_argument('--inver_param', default=0.001, type=float, help='Hyper_param for inverse_pred')
    parser.add_argument('--recon_param', default=0.001, type=float, help='Hyper_param for forward_pred')
    parser.add_argument('--with_info', default=False, action='store_true')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    resnet_depth = args.resnet_depth
    trial = args.trial
    
    recon_param = args.recon_param
    inver_param = args.inver_param

    # data prepare
    if args.dataset == 'cifar':
        train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    else:
        # our self-supervised signal construction strategy
        train_data = utils.Our_Omniglot(root='data', background=True, transform=utils.omniglot_train_transform, 
                                character_target_transform=None, alphabet_target_transform=None, download=True, 
                                contrast_training=True)
        # self-supervised signal construction strategy in SimCLR
        #train_data = utils.Our_Omniglot_v2(root='data', background=True, transform=utils.omniglot_train_transform, 
        #                        character_target_transform=None, alphabet_target_transform=None, download=True, 
        #                        contrast_training=True)
    
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    
    if args.dataset == 'cifar':
        memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    else:
        memory_data = utils.Our_Omniglot(root='data', background=False, transform=utils.omniglot_test_transform,
                                         character_target_transform=None, alphabet_target_transform=None, download=True,
                                         eval_split_train=True, out_character=True, contrast_training=False)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    
    if args.dataset == 'cifar':
        test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    else:
        test_data = utils.Our_Omniglot(root='data', background=False, transform=utils.omniglot_test_transform,
                                         character_target_transform=None, alphabet_target_transform=None, download=True,
                                         eval_split_train=False, out_character=True, contrast_training=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # calculating information
    if args.with_info:
        info_models, info_optimizer = init_models('Feature_Information')
        info_results = {'I(Z;S)': [], 'I(Z;T)': [], 'H(Z|T)': [], 'H(Z|S)': []}
        info_dct = {
            'info_models': info_models,
            'info_optimizer': info_optimizer,
            'info_results': info_results,
        }
    else:
        info_dct = None
        
    # model setup and optimizer config
    if args.dataset == 'cifar':
        model = Model(feature_dim, resnet_depth=resnet_depth).cuda()
        recon_model = None
    else:
        model = Omniglot_Model().cuda()
        recon_model = Recon_Omniglot_Model().cuda() if args.loss_type >= 3 else None
        
    if args.dataset == 'cifar':
        flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    else:
        flops, params = profile(model, inputs=(torch.randn(1, 1, 28, 28).cuda(),))
        #flops, params = profile(model, inputs=(torch.randn(1, 1, 56, 56).cuda(),))
        #flops, params = profile(model, inputs=(torch.randn(1, 1, 105, 105).cuda(),))
        if recon_model is not None:
            recon_flops, recon_params = profile(recon_model, inputs=(torch.randn(1, 1024).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    if recon_model is not None:
        recon_flops, recon_params = clever_format([recon_flops, recon_params])
        print('# Recon_Model Params: {} FLOPs: {}'.format(recon_params, recon_flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    recon_optimizer = optim.Adam(recon_model.parameters(), lr=1e-3, weight_decay=1e-6) if recon_model is not None \
                        else None
    #optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-6)
    #milestone1, milestone2 = int(args.epochs*0.4), int(args.epochs*0.7)
    #lr_decay = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone1, milestone2], gamma=0.1)
    if args.dataset == 'cifar':
        c = len(memory_data.classes)
    else:
        c = 659 #c = 20

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    if args.dataset == 'cifar':
        save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, resnet_depth, feature_dim, temperature, k, \
                                                          batch_size, epochs)
    else:
        save_name_pre = '{}_{}_{}_{}_{}'.format(args.dataset, args.loss_type, recon_param, inver_param, trial)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        if(epoch%4)==1:
            train_loss = train(epoch, model, train_loader, optimizer, recon_model, args.loss_type, recon_optimizer, info_dct)
        else:
            train_loss = train(epoch, model, train_loader, optimizer, recon_model, args.loss_type, recon_optimizer, None)
        #lr_decay.step()
        results['train_loss'].append(train_loss)
        
        if args.dataset == 'cifar':
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        else:
            test_acc_1, test_acc_5 = omniglot_test(model, memory_loader, test_loader)
            
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        
        if (epoch%4)==1 and info_dct is not None:
            info_data_frame = pd.DataFrame(data=info_dct['info_results'], index=range(1, epoch + 1, 4))
            if args.loss_type==1:
                info_data_frame.to_csv('results/Feature_Information.csv', index_label=info_dct['epoch'])
            elif args.loss_type==2:
                info_data_frame.to_csv('results/Feature_Information_min_H.csv', index_label=info_dct['epoch'])
        
        if args.dataset == 'cifar':
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
