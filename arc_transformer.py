import torch
import numpy as np
from ARCDataset import ARCTrain
import argparse
import os
import torch.nn.functional as F
from copy import deepcopy
from transformer_model import TransformerModel
from utils import seed_everything

seed_everything()


# does one shot validation without any training
# just to get a very rough idea to monitor progress
@torch.no_grad()
def validation(model, arc_dataset):
    model.eval()
    outputs = F.softmax(model(arc_dataset.val_x_batch), dim=1)
    idx = torch.where(arc_dataset.val_y_batch != 10)[0]
    outputs = outputs.argmax(2).reshape(-1,)[idx]
    acc = (outputs ==
           arc_dataset.val_y_batch[idx]).float().mean().item()
    return np.mean(acc)


def main():
    weights_dir = './model_weights'
    os.makedirs(weights_dir, exist_ok=True)
    print(args)
    ntokens = 11  # the size of vocabulary
    emsize = 32  # embedding dimension
    nhid = 64  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # the number of heads in the multiheadattention models
    dropout = 0.5  # the dropout value
    device = 'cuda'

    innerstepsize = 1e-2  # stepsize in inner loop
    innerepochs = 50  # number of epochs of each inner loop

    outerstepsize = 0.1
    # ntoken, ninp, nhead, nhid, nlayers, dropout=0.5
    model = TransformerModel(ntokens, emsize, nhead,
                             nhid, nlayers, dropout).to(device)

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    arc_dataset = ARCTrain(root='/home/sid/Desktop//arc/data/',
                           imgsz=args.imgsz)
    arc_dataset.val_x_batch = arc_dataset.val_x_batch.reshape(
        -1, args.imgsz*args.imgsz).long()
    best_val_acc = 0
    task_ids = np.arange(len(arc_dataset))
    for epoch in range(1, args.epoch+1):
        # randomly shuffle the task orders.
        np.random.shuffle(task_ids)
        for step, task_num in enumerate(task_ids):
            x, y = arc_dataset[task_num]
            model.train()
            train_acc = []
            train_losses = []
            x = x.long().to(device).reshape(-1, args.imgsz*args.imgsz)
            y = y.to(device).long()

            weights_before = deepcopy(model.state_dict())
            optimizer = torch.optim.AdamW(model.parameters(), lr=innerstepsize)
            # starts training over the task
            for _ in range(innerepochs):
                optimizer.zero_grad()
                outputs = model(x).reshape(-1, args.num_class)
                loss = F.cross_entropy(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                train_losses.append(loss.item())
                acc = (outputs.argmax(1) == y).float().mean().item()
                train_acc.append(acc)

            valid_before_acc = validation(model, arc_dataset)
            # if (step % 20 == 0):
            #     outerstepsize = outerstepsize * \
            #         (1 - epoch / args.epoch)  # linear schedule
            #     print('outerstepsize:', outerstepsize)

            # print('Interpolating weights.')
            # Interpolate between current weights and trained weights from this task
            # I.e. (weights_before - weights_after) is the meta-gradient
            weights_after = model.state_dict()
            model.load_state_dict({name:
                                   weights_before[name] + (weights_after[name] -
                                                           weights_before[name]) * outerstepsize
                                   for name in weights_before})

            valid_after_acc = validation(model, arc_dataset)
            print('epoch:', epoch, 'step:', step, '\ttraining loss:',
                  f'{np.mean(train_losses):.3}', '\ttraining acc:',
                  f'{np.mean(train_acc):.3}', '\tvalidation before acc:',
                  f'{np.mean(valid_before_acc):.3}', '\tvalidation after acc:',
                  f'{np.mean(valid_after_acc):.3}')
            if valid_after_acc > best_val_acc:  # evaluation
                best_val_acc = np.copy(valid_after_acc)
                fn = f'./{weights_dir}/epoch_{epoch}_step_{step}_acc_{valid_after_acc:.3}.pth'
                torch.save(model.state_dict(), fn)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int,
                           help='epoch number', default=50)
    argparser.add_argument('--num_class', type=int,
                           help='number of classes', default=11)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=15)
    args = argparser.parse_args()

    main()
