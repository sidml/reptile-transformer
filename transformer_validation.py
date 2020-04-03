import torch
import numpy as np
from ARCDataset import ARCVal
import argparse
import os
import torch.nn.functional as F
from transformer_model import TransformerModel
from glob import glob
from utils import plot_figure, seed_everything

seed_everything()


def main():
    os.makedirs('./model_preds', exist_ok=True)

    print(args)

    ntokens = 11  # the size of vocabulary
    emsize = 32  # embedding dimension
    nhid = 64  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # the number of heads in the multiheadattention models
    dropout = 0.5  # the dropout value
    device = 'cuda'

    print('ntokens', ntokens, 'emsize', emsize, 'nhid', nhid)
    print('nlayers', nlayers, 'nhead', nhead, 'dropout', dropout)
    print()

    innerstepsize = 1e-2  # stepsize in inner SGD
    innerepochs = 100  # number of epochs of each inner SGD
    # ntoken, ninp, nhead, nhid, nlayers, dropout=0.5
    model = TransformerModel(ntokens, emsize, nhead,
                             nhid, nlayers, dropout).to(device)

    # batchsz here means total episode number
    arc_dataset = ARCVal(
        root='/home/sid/Desktop/arc/data/', imgsz=args.imgsz)

    def cond(x): return float(x.split('/')[-1].split('_')[-1][:-4])
    all_model_fn = sorted(glob('./model_weights/*.pth'), key=cond)[-3:]
    # state = torch.load(all_model_fn[0])
    # print('averaging weights of')
    # for fn in all_model_fn[1:]:
    #     print(fn)
    #     for name in state:
    #         state[name] = state[name] + (state[name] - state[name])
    # model.load_state_dict(state)
    # all_model_fn = ['./models/reptile_sz30_epoch_0_step_139_acc_0.299.pth']
    for fn in all_model_fn:
        all_val_acc, all_train_acc = [], []
        print('Processing fn', fn)
        for step, (train_x, train_y, val_x, val_y) in enumerate(arc_dataset):
            state = torch.load(fn)
            model.load_state_dict(state)

            optimizer = torch.optim.AdamW(model.parameters(), lr=innerstepsize)

            train_x = train_x.to(device).reshape(-1, args.imgsz*args.imgsz)
            val_x = val_x.to(device).reshape(-1, args.imgsz*args.imgsz)
            train_y = train_y.to(device)
            val_y = val_y.to(device)

            train_losses = []
            train_acc = []
            model.train()
            for _ in range(innerepochs):
                optimizer.zero_grad()
                outputs = model(train_x).reshape(-1, args.num_class)
                loss = F.cross_entropy(outputs, train_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                acc = (outputs.argmax(1) == train_y).float().mean().item()
                train_acc.append(acc)

            all_train_acc.append(np.mean(train_acc))
            model.eval()
            with torch.no_grad():
                outputs = F.softmax(model(val_x), dim=1)
                outputs = outputs.argmax(2).reshape(-1, args.imgsz, args.imgsz)
                val_acc = (outputs == val_y).float().mean().item()
                plot_figure(train_x, train_y, val_x, outputs,
                            im_num=step, img_sz=args.imgsz)
            print('training loss:',
                  np.mean(train_losses), '\ttraining acc:', np.mean(train_acc),
                  '\tvalidation acc:', val_acc)
            all_val_acc.append(val_acc)

        print('mean train acc:', np.mean(all_train_acc),
              'stddev train acc:', np.std(all_train_acc))

        print(f'mean val acc: {np.mean(all_val_acc):.3}',
              f'stddev val acc: {np.std(all_val_acc):.3}', 'max val acc:',
              f'{max(all_val_acc):.3}', 'min val acc:', f'{min(all_val_acc):.3}',
              'num complete correct:', (np.array(all_val_acc) == 1).sum())
        print()


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_class', type=int,
                           help='number of classes', default=11)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=15)

    args = argparser.parse_args()

    main()
