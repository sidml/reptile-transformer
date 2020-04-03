import torch
import numpy as np
from ARCDataset import ARCTest
import argparse
import os
import torch.nn.functional as F
from matplotlib import colors
from glob import glob
from transformer_model import TransformerModel
from utils import seed_everything, plot_figure

seed_everything()

innerstepsize = 1e-2  # stepsize in inner SGD
innerepochs = 50  # number of epochs of each inner SGD

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF'])
norm = colors.Normalize(vmin=0, vmax=10)


def main():
    weights_dir = './model_weights'
    os.makedirs('./model_preds', exist_ok=True)

    print(args)
    ntokens = 11  # the size of vocabulary
    emsize = 32  # embedding dimension
    nhid = 64  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # the number of heads in the multiheadattention models
    dropout = 0.5  # the dropout value
    device = torch.device('cuda')

    model = TransformerModel(ntokens, emsize, nhead,
                             nhid, nlayers, dropout).to(device)

    def cond(x): return float(x.split('/')[-1].split('_')[-1][:-4])
    all_model_fn = sorted(glob(f'./{weights_dir}/*.pth'), key=cond)[-1]
    print('Using model weights from', all_model_fn)

    # batchsz here means total episode number
    arc_dataset = ARCTest(
        root='/home/sid/Desktop/arc/data/', imgsz=args.imgsz)

    all_train_acc = []
    for step, ((x, y), q) in enumerate(zip(arc_dataset, arc_dataset.query_x_batch)):
        # print('step:', step)
        state = torch.load(all_model_fn)
        model.load_state_dict(state)

        optimizer = torch.optim.AdamW(model.parameters(), lr=innerstepsize)
        x, y = x.to(device), y.to(device)
        x = x.to(device).reshape(-1, args.imgsz*args.imgsz).long()

        train_losses = []
        train_acc = []
        model.train()
        for _ in range(innerepochs):
            optimizer.zero_grad()
            outputs = model(x).reshape(-1, args.num_class)
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            acc = (outputs.argmax(1) == y).float().mean().item()
            train_acc.append(acc)
        print('\ttraining loss:',
              np.mean(train_losses), '\ttraining acc:', np.mean(train_acc))

        all_train_acc.append(np.mean(train_acc))
        model.eval()
        with torch.no_grad():
            q = torch.tensor(
                q.reshape(-1, args.imgsz*args.imgsz)).to(device).long()
            # print(q.shape)
            outputs = F.softmax(model(q), dim=1)
            outputs = outputs.argmax(2).reshape(-1, args.imgsz, args.imgsz)
            plot_figure(x, y, q, outputs, im_num=step, img_sz=args.imgsz)

    print('\nmean train acc:', np.mean(all_train_acc),
          'stddev train acc:', np.std(all_train_acc))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int,
                           help='epoch number', default=501)
    argparser.add_argument('--num_class', type=int,
                           help='number of classes', default=11)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=15)

    args = argparser.parse_args()

    main()
