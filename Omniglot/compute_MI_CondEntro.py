import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50
import math
import utils
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import pandas as pd


# ## Utility Functions

class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
def mlp(dim, hidden_dim, output_dim, layers=1, batch_norm=False):
    if batch_norm:
        seq = [nn.Linear(dim, hidden_dim), nn.BatchNorm1d(num_features=hidden_dim),\
               nn.ReLU(inplace=True)]
        for _ in range(layers):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(num_features=hidden_dim),\
                    nn.ReLU(inplace=True)]
    else:
        seq = [nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(layers):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


def init_models(stage, additional_encoder=False):
    if stage == 'AE':
        norm_encoder = Encoder(normalize=True).cuda()
        norm_decoder = Decoder().cuda()
        encoder = Encoder(normalize=False).cuda()
        decoder = Decoder().cuda()
        
        params = list(norm_encoder.parameters()) + list(norm_decoder.parameters()) +\
                 list(encoder.parameters()) + list(decoder.parameters())

        norm_encoder.train()
        norm_decoder.train()
        encoder.train()
        decoder.train()
        
        models = {
            'norm_encoder': norm_encoder,
            'norm_decoder': norm_decoder,
            'encoder': encoder,
            'decoder': decoder,
        }
    elif stage == 'Raw_Information' or stage == 'Feature_Information':
        if stage == 'Raw_Information':
            norm_encoder = Encoder(normalize=True).cuda()
            norm_encoder.load_state_dict(torch.load('results/norm_encoder.pth'))
            norm_encoder.eval()
            if additional_encoder:
                feat_encoder = Encoder(normalize=True).cuda()
                feat_encoder.load_state_dict(torch.load('results/norm_encoder.pth'))
                feat_encoder.train()
                params = list(feat_encoder.parameters())
            else:
                feat_encoder = None
                params = []
        else:
            norm_encoder = None
            feat_encoder = None
            params = []
            
        encoder = Encoder(normalize=False).cuda()
        encoder.load_state_dict(torch.load('results/encoder.pth'))
        
        mi_z_z_model = MI_Z_Z_Model().cuda()
        mi_z_t_model = MI_Z_T_Model().cuda()
        cond_z_t_model = Cond_Z_T_Model().cuda()
        cond_z_z_model = Cond_Z_Z_Model().cuda()
        
        params = params + list(mi_z_z_model.parameters()) + list(mi_z_t_model.parameters()) +\
                 list(cond_z_t_model.parameters()) + list(cond_z_z_model.parameters()) 
        
        encoder.eval()
        mi_z_z_model.train()
        mi_z_t_model.train()
        cond_z_t_model.train()
        cond_z_z_model.train()
        
        models = {
            'encoder': encoder,
            'norm_encoder': norm_encoder,
            'feat_encoder': feat_encoder,
            'mi_z_z_model': mi_z_z_model,
            'mi_z_t_model': mi_z_t_model,
            'cond_z_t_model': cond_z_t_model,
            'cond_z_z_model': cond_z_z_model,
        }
    optimizer = optim.Adam(params, lr=1e-3)
    
    return models, optimizer


# ## Auto-Encoding Structure (for one-to-one mapping)

class Encoder(nn.Module):
    def __init__(self, normalize=False):
        super(Encoder, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 28
            #nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 28
            #nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 14
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 14
            #nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 7
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 7
            #nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 3
            nn.Flatten(),
            nn.Linear(9*128, 1024),
        )
        self.normalize = normalize

    def forward(self, _input):
        feature = self.f(_input)
        if self.normalize:
            return F.normalize(feature, dim=-1)
        else:
            return feature
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(1024, 9*128, bias=False),
            #nn.BatchNorm1d(num_features=9*128),
            nn.ReLU(inplace=True), # (9*128 -> 3*3*128)
            Lambda(lambda x: x.view(-1, 128, 3, 3)),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0,
                              output_padding=0, bias=False), # out: 7
            #nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1,
                              output_padding=1, bias=False), # out: 14
            #nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1,
                              output_padding=1, bias=False), # out: 28
            #nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1,
                              output_padding=0, bias=True), # out: 28
        )

    def forward(self, _input):
        return self.f(_input)


# ## Infomation Functions

class MI_Z_T_Model(nn.Module):
    def __init__(self):
        super(MI_Z_T_Model, self).__init__()
        self._g = mlp(1024, 512, 512)
        self._h = mlp(964, 512, 512)

    def forward(self, z, t):
        t = F.one_hot(t, num_classes=964)
        
        scores = torch.matmul(self._g(z), self._h(t.float()).t())
        return scores


class MI_Z_Z_Model(nn.Module):
    def __init__(self):
        super(MI_Z_Z_Model, self).__init__()
        self._g = mlp(1024, 512, 512)
        self._h = mlp(1024, 512, 512)

    def forward(self, z1, z2):
        scores = torch.matmul(self._g(z1), self._h(z2).t())
        return scores


class Cond_Z_T_Model(nn.Module):
    def __init__(self):
        super(Cond_Z_T_Model, self).__init__()
        self._g = mlp(964, 512, 1024)

    def forward(self, t):
        t = F.one_hot(t, num_classes=964)
        recon_z = self._g(t.float())
        return recon_z


class Cond_Z_Z_Model(nn.Module):
    def __init__(self):
        super(Cond_Z_Z_Model, self).__init__()
        self._g = mlp(1024, 512, 1024)

    def forward(self, z):
        recon_z = self._g(z)
        return recon_z


# ## Training

def AE_loss(x, s, encoder, decoder):
    zx = encoder(x)
    zs = encoder(s)
    hat_x = decoder(zx)
    hat_s = decoder(zs)
    return F.binary_cross_entropy_with_logits(hat_x, x) +\
           F.binary_cross_entropy_with_logits(hat_s, s)
def Enc_z(x, s, encoder):
    return encoder(x), encoder(s)
def AE_Step(pos_1, pos_2, models, optimizer):
    norm_encoder, norm_decoder = models['norm_encoder'], models['norm_decoder']
    encoder, decoder = models['encoder'], models['decoder']
    
    optimizer.zero_grad()
    loss = 0.5*AE_loss(pos_1, pos_2, norm_encoder, norm_decoder) +\
           0.5*AE_loss(pos_1, pos_2, encoder, decoder)
    loss.backward()
    optimizer.step()
    
    return loss.item()


# Maximization
def MI_Estimator(ft1, ft2, model):
    '''
    ft1, ft2: r.v.s
    model: takes ft1 and ft2, output batch_size x batch_size
    '''
    scores = model(ft1, ft2)
    
    # optimal critic f(ft1, ft2) = log {p(ft1, ft2)/p(ft1)p(ft2)}
    def js_fgan_lower_bound_obj(scores):
        """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
        scores_diag = scores.diag()
        first_term = -F.softplus(-scores_diag).mean()
        n = scores.size(0)
        second_term = (torch.sum(F.softplus(scores)) -
                       torch.sum(F.softplus(scores_diag))) / (n * (n - 1.))
        return first_term - second_term
    # if the input is in log form
    def direct_log_density_ratio_mi(scores):
        return scores.diag().mean()   
    
    train_val = js_fgan_lower_bound_obj(scores)
    eval_val = direct_log_density_ratio_mi(scores)
    
    with torch.no_grad():
        eval_train = eval_val - train_val
        
    return train_val + eval_train


# Minimization
def Conditional_Entropy(ft1, ft2, model):
    '''
    Calculating H(ft2|ft1) by min_Q H[P(ft2|ft1), Q(ft2|ft1)]
    ft1: discrete or continuous
    ft2: continuous (k-dim.)
    We assume Q(ft2|ft1) is Gaussian.
    model (Q): takes ft1, out the reconstructed ft2
    '''
    hat_ft2 = model(ft1)

    # sigma = l2_norm of ft2 (we let it be 1)
    # when Q = Normal(mu(ft1), sigma^2I) -> -logQ = log(sqrt((2*pi)^k sigma^(2k))) + 
    #            0.5*1/(sigma^2)*(y-mu)^T(y-mu)
    # H[P(ft2|(ft1), Q(ft2|(ft1)] = E_{P_{(ft1,ft2}} [-logQ]
    dim = ft2.shape[1]
    bsz = ft2.shape[0]
    
    #cond_entropy = 0.5*dim*math.log(2*math.pi) + 0.5*(F.mse_loss(hat_ft2, ft2, reduction='sum')/bsz)
    #return cond_entropy
    scaled_cond_entropy = F.mse_loss(hat_ft2, ft2, reduction='sum')/bsz
    return scaled_cond_entropy



def Information_Step(pos_1, pos_2, t, models, optimizer, zx=None, zs=None):    
    feat_encoder, norm_encoder, encoder,\
    mi_z_z_model, mi_z_t_model, cond_z_t_model, cond_z_z_model =\
        models['feat_encoder'], models['norm_encoder'], models['encoder'], models['mi_z_z_model'],\
        models['mi_z_t_model'], models['cond_z_t_model'], models['cond_z_z_model']

    ae_zx, ae_zs = Enc_z(pos_1, pos_2, encoder)
    if zx is None and zs is None:
        if feat_encoder is not None:
            I_zx, I_zs = Enc_z(pos_1, pos_2, feat_encoder)
        else:
            I_zx, I_zs = Enc_z(pos_1, pos_2, norm_encoder)
        H_zx, H_zs = Enc_z(pos_1, pos_2, norm_encoder)
    else:
        I_zx, I_zs = zx, zs
        H_zx, H_zs = zx, zs
        
    I_Z_T = 0.5*MI_Estimator(I_zx, t, mi_z_t_model) +\
            0.5*MI_Estimator(I_zs, t, mi_z_t_model)
    
    I_Z_S = 0.5*MI_Estimator(I_zx, ae_zs, mi_z_z_model) +\
                0.5*MI_Estimator(I_zs, ae_zx, mi_z_z_model)
    
    H_Z_T = 0.5*Conditional_Entropy(t, H_zx, cond_z_t_model) +\
            0.5*Conditional_Entropy(t, H_zs, cond_z_t_model)
    
    H_Z_S = 0.5*Conditional_Entropy(ae_zs, H_zx, cond_z_z_model) +\
            0.5*Conditional_Entropy(ae_zx, H_zs, cond_z_z_model)
    
    optimizer.zero_grad()
    loss = -I_Z_S - I_Z_T  + H_Z_T + H_Z_S
    loss.backward()
    optimizer.step()
    
    return I_Z_S.item(), I_Z_T.item(), H_Z_T.item(), H_Z_S.item()


# ## Script for Calculating  I(X;S), I(X;S|T), I(X;T), H(X|T), H(X|S)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=482, type=int, help='Number of images in each mini-batch\
                         (964/2=482 for omniglot and 512 for cifar)')
    parser.add_argument('--epochs', default=23000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--stage', default='AE', type=str, help='AE or Raw_Information')
    parser.add_argument('--additional_encoder', default=False, action='store_true')

    # args parse
    args = parser.parse_args()
    batch_size, epochs, stage, additional_encoder = args.batch_size, args.epochs, args.stage, args.additional_encoder
    
    train_data = utils.Our_Omniglot(root='data', background=True, transform=utils.omniglot_train_transform, 
                                character_target_transform=None, alphabet_target_transform=None, download=True, 
                                contrast_training=True)
    
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)

    # model setup and optimizer config
    models, optimizer = init_models(stage, additional_encoder)
    
    if stage == 'Raw_Information':
        results = {'I(X;S)': [], 'I(X;T)': [], 'H(X|T)': [], 'H(X|S)': []}
    
    for epoch in range(1, epochs + 1):
        train_bar = tqdm(train_loader)
        total_num = 0
        if stage == 'Raw_Information':
            I_X_S_total, I_X_T_total, H_X_T_total, H_X_S_total =\
                0.0, 0.0, 0.0, 0.0
        elif stage == 'AE':
            AE_Loss_total = 0.0
        
        for pos_1, pos_2, target in train_bar:
            pos_1, pos_2, target = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True),\
                                   target.cuda(non_blocking=True)
            total_num += 1
            if stage == 'Raw_Information':
                I_X_S, I_X_T, H_X_T, H_X_S = \
                    Information_Step(pos_1, pos_2, target, models, optimizer, zx=None, zs=None)
                
                I_X_S_total += I_X_S
                I_X_T_total += I_X_T
                H_X_T_total += H_X_T
                H_X_S_total += H_X_S
            elif stage == 'AE':
                AE_Loss_total += AE_Step(pos_1, pos_2, models, optimizer)

        if stage == 'Raw_Information':
            print('Epoch: {}, I(X;S): {}, I(X;T): {}, H(X|T): {}, H(X|S): {}'\
                   .format(epoch, I_X_S_total / total_num, I_X_T_total / total_num,\
                           H_X_T_total / total_num, H_X_S_total / total_num))
            
            results['I(X;S)'].append(I_X_S_total / total_num)
            results['I(X;T)'].append(I_X_T_total / total_num)
            results['H(X|T)'].append(H_X_T_total / total_num)
            results['H(X|S)'].append(H_X_S_total / total_num)
            
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            
            if additional_encoder:
                data_frame.to_csv('results/Raw_Information_additional_encoder.csv', index_label='epoch')
            else:
                data_frame.to_csv('results/Raw_Information.csv', index_label='epoch')
        elif stage == 'AE':
            print('Epoch: {}, AE_Loss: {}'.format(epoch, AE_Loss_total / total_num))
            
            # save encoder
            torch.save(models['norm_encoder'].state_dict(), 'results/norm_encoder.pth')
            torch.save(models['encoder'].state_dict(), 'results/encoder.pth')


def information(epoch, train_loader, inner_epochs, net, models, optimizer):
    net.eval()
    I_Z_S_total, I_Z_T_total, H_Z_T_total, H_Z_S_total =\
        0.0, 0.0, 0.0, 0.0
    total_num = 0
    
    for _in in range(inner_epochs):
        train_bar = tqdm(train_loader)
        for pos_1, pos_2, target in train_bar:
            pos_1, pos_2, target = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True),\
                                   target.cuda(non_blocking=True)
            total_num += 1
            zx, _ = net(pos_1)
            zs, _ = net(pos_2)
            I_Z_S, I_Z_T, H_Z_T, H_Z_S = \
                Information_Step(pos_1, pos_2, target, models, optimizer, zx=zx, zs=zs)
            
            I_Z_S_total += I_Z_S
            I_Z_T_total += I_Z_T
            H_Z_T_total += H_Z_T
            H_Z_S_total += H_Z_S

            print('Epoch: {}, Inner Epoch: {}, I(Z;S): {}, I(Z;T): {}, H(Z|T): {}, H(Z|S): {}'\
               .format(epoch, _in, I_Z_S, I_Z_T, H_Z_T, H_Z_S))
            
    return I_Z_S_total/total_num , I_Z_T_total/total_num, H_Z_T_total/total_num,\
           H_Z_S_total/total_num
