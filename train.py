import ast
import argparse
import math
import time
import torch
import torch.nn as nn
from net import gtnet
from net_lstm import net_lstm
import numpy as np
import importlib
import random
from o_util import *
from trainer import Optim
import sys
from random import randrange
from matplotlib import pyplot as plt
import time
from safetensors.torch import save_file

plt.rcParams['savefig.dpi'] = 1200

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        #temp_X=X
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

            id = torch.tensor(id).to(device)
            tx = X[:, :, :, :] #id was in third colum
            ty = Y[:, :, :] #id was in third colum
            output = model(tx)#id was in second arg            
            output = torch.squeeze(output,3)
            scale = data.scale.expand(output.size(0), output.size(1), data.m)
            scale = scale[:,:,:] #id was in third colum
            
            output*=scale #by Zaid
            ty*=scale
            
            # #inverse diff
            # output=inverse_diff_3d(output,I,data.shift)
            # ty=inverse_diff_3d(ty,I,data.shift)

            loss = criterion(output, ty)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * data.m)
            
            # perform gradient clipping
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            grad_norm = optim.step()

        if iter%1==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * output.size(1)* data.m)))
        iter += 1
    return total_loss / n_samples


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='/content/data/sm_data.txt',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/Bayesian/o_model.safetensors',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:1',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=142,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=10,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=36,help='output sequence length')
parser.add_argument('--horizon', type=int, default=1) 
parser.add_argument('--layers',type=int,default=5,help='number of layers')
parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=10,help='clip')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')


args = parser.parse_args()
device = torch.device('cpu') #modified!!!!!!!!!!!!!!!!!!!!!
torch.set_num_threads(3)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fixed_seed = 123
set_random_seed(fixed_seed)


#read hyper-parameters
filename = "model/Bayesian/hp.txt"
with open(filename, 'r') as file:
    content = file.read()
    hp = ast.literal_eval(content)

print('hp',hp)

#train the model
gcn_depth=hp[0]
lr=hp[1]
conv=hp[2]
res=hp[3]
skip=hp[4]
end=hp[5]
layer=hp[-2]
k=hp[6]
dropout=hp[7]
dilation_ex=hp[8]
node_dim=hp[9]
prop_alpha=hp[10]
tanh_alpha=hp[11]
epochs=hp[-1]


Data = DataLoaderS(args.data, 0.43, 0.30, device, args.horizon, args.seq_in_len, args.normalize,args.seq_out_len)
   

print('train X:',Data.train[0].shape)
print('train Y:', Data.train[1].shape)
print('valid X:',Data.valid[0].shape)
print('valid Y:',Data.valid[1].shape)
print('test X:',Data.test[0].shape)
print('test Y:',Data.test[1].shape)
print('test window:', Data.test_window.shape)

print('length of training set=',Data.train[0].shape[0])#Zaid
print('length of validation set=',Data.valid[0].shape[0])#Zaid
print('length of testing set=',Data.test[0].shape[0])#Zaid
print('valid=',int((0.43 + 0.3) * Data.n))



model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
            device, Data.adj, dropout=dropout, subgraph_size=k,
            node_dim=node_dim, dilation_exponential=dilation_ex,
            conv_channels=conv, residual_channels=res,
            skip_channels=skip, end_channels= end,
            seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
            layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)



print(args)
print('The recpetive field size is', model.receptive_field)
nParams = sum([p.nelement() for p in model.parameters()])
print('Number of model parameters is', nParams, flush=True)

if args.L1Loss:
    criterion = nn.L1Loss(reduction='sum').to(device)
else:
    criterion = nn.MSELoss(reduction='sum').to(device)
evaluateL2 = nn.MSELoss(reduction='sum').to(device) #MSE
evaluateL1 = nn.L1Loss(reduction='sum').to(device) #MAE


optim = Optim(
    model.parameters(), args.optim, lr, args.clip, lr_decay=args.weight_decay
)

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('begin training')
    for epoch in range(1, epochs + 1):
        print('epoch:',epoch)
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
    save_file(model.state_dict(), args.save)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')