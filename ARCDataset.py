import torch
from torch.utils.data import Dataset
import numpy as np
import json
from glob import glob
from utils import create_batch


class ARCTrain(Dataset):

    def __init__(self, root, imgsz=15):
        super(ARCTrain, self).__init__()
        self.out_rows, self.out_cols = imgsz, imgsz
        task_paths = f'{root}/training/*.json'
        train_x_batch, train_y_batch, val_x_batch,\
            val_y_batch = create_batch(
                glob(task_paths), self.out_rows, self.out_cols)

        task_paths = glob(f'{root}/evaluation/*.json')
        test_task_ids = list(map(lambda x: x.split(
            '/')[-1], glob(f'{root}/test/*.json')))
        task_paths = [tp for tp in task_paths if tp.split(
            '/')[-1] not in test_task_ids]

        self.train_x_batch, self.train_y_batch, self.val_x_batch,\
            self.val_y_batch = create_batch(
                task_paths, self.out_rows, self.out_cols)

        for q1, q2, q3, q4 in zip(train_x_batch, train_y_batch,
                                  val_x_batch, val_y_batch):
            self.train_x_batch.append(q1)
            self.train_y_batch.append(q2)
            self.val_x_batch.append(q3)
            self.val_y_batch.append(q4)

        self.val_x_batch = np.vstack(np.array(self.val_x_batch))[:, None]
        self.val_y_batch = np.vstack(np.array(self.val_y_batch))[:, None]
        self.val_x_batch = torch.tensor(self.val_x_batch).float().cuda()
        self.val_y_batch = torch.tensor(
            self.val_y_batch).float().reshape(-1,).cuda()

        print('Number of training tasks', len(self.train_x_batch))
        print('Number of validation tasks', len(self.val_x_batch))

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        """
        train_x = torch.tensor(
            self.train_x_batch[index], dtype=torch.long)
        train_y = torch.tensor(
            self.train_y_batch[index], dtype=torch.long)
        return train_x[:, None], train_y.reshape(-1)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        # return self.batchsz
        return len(self.train_x_batch)


class ARCVal(Dataset):

    def __init__(self, root, imgsz=30):

        super(ARCVal, self).__init__()
        self.out_rows, self.out_cols = imgsz, imgsz
        task_paths = glob(f'{root}/evaluation/*.json')
        test_task_ids = list(map(lambda x: x.split(
            '/')[-1], glob(f'{root}/test/*.json')))
        task_paths = [tp for tp in task_paths if tp.split(
            '/')[-1] in test_task_ids]
        self.train_x_batch, self.train_y_batch, self.val_x_batch,\
            self.val_y_batch = create_batch(task_paths, self.out_rows,
                                            self.out_cols)

        print('Number of training tasks', len(self.train_x_batch))
        print('Number of validation tasks', len(self.val_x_batch))

    def __getitem__(self, index):
        train_x = torch.tensor(
            self.train_x_batch[index], dtype=torch.long)
        train_y = torch.tensor(
            self.train_y_batch[index], dtype=torch.long)

        val_x = torch.tensor(
            self.val_x_batch[index], dtype=torch.long)
        val_y = torch.tensor(
            self.val_y_batch[index], dtype=torch.long)
        return train_x[:, None], train_y.reshape(-1), val_x, val_y

    def __len__(self):
        return len(self.train_x_batch)


class ARCTest(Dataset):

    def __init__(self, root, imgsz=30):
        super(ARCTest, self).__init__()
        self.out_rows, self.out_cols = imgsz, imgsz
        task_paths = f'{root}/test/*.json'
        self.train_x_batch, self.train_y_batch, self.query_x_batch = self.create_batch(
            sorted(glob(task_paths)))

    def pad_im(self, task, out_rows, out_cols, cval=10):

        ip = []
        op = []
        for mode in ['train']:
            num_pairs = len(task[mode])
            input_im = np.zeros((num_pairs, 1, out_rows, out_cols))
            output_im = np.zeros(
                (num_pairs, 1, out_rows, out_cols), dtype=np.long)
            for task_num in range(num_pairs):
                im = np.array(task[mode][task_num]['input'])
                nrows, ncols = im.shape
                if (nrows > out_rows) or (ncols > out_cols):
                    return 0, 0, 1, 0
                im = np.pad(im, ((out_rows-nrows, 0), (out_cols-ncols, 0)), mode='constant',
                            constant_values=(cval, cval))

                input_im[task_num, 0] = im
                im = np.array(task[mode][task_num]['output'])
                nrows, ncols = im.shape
                if (nrows > out_rows) or (ncols > out_cols):
                    return 0, 0, 1, 0
                im = np.pad(im, ((out_rows-nrows, 0), (out_cols-ncols, 0)), mode='constant',
                            constant_values=(cval, cval))
                output_im[task_num, 0] = im
            ip.extend(input_im)
            op.extend(output_im)

            test_ip = []
            num_pairs = len(task['test'])
            input_im = np.zeros((num_pairs, 1, out_rows, out_cols))
            for task_num in range(num_pairs):
                im = np.array(task['test'][task_num]['input'])
                nrows, ncols = im.shape
                if (nrows > out_rows) or (ncols > out_cols):
                    return 0, 0, 1, 0
                im = np.pad(im, ((out_rows-nrows, 0), (out_cols-ncols, 0)), mode='constant',
                            constant_values=(cval, cval))

                input_im[task_num, 0] = im
            test_ip.extend(input_im)

        return np.vstack(ip), np.vstack(op), 0, np.vstack(test_ip)

    def create_batch(self, task_paths):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        x_batch = []  # train set batch
        y_batch = []  # train set batch
        query_x_batch = []
        for task_file in task_paths:
            with open(task_file, 'r') as f:
                task = json.load(f)
            input_im, output_im, not_valid, query_im = self.pad_im(task, self.out_rows,
                                                                   self.out_cols)
            if not_valid:
                continue
            x_batch.extend(input_im[None])
            y_batch.extend(output_im[None])
            query_x_batch.extend(query_im[None])
        return x_batch, y_batch, query_x_batch

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        """

        train_x = torch.tensor(
            self.train_x_batch[index], dtype=torch.float32)
        train_y = torch.tensor(
            self.train_y_batch[index], dtype=torch.long)
        return train_x[:, None], train_y.reshape(-1)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        # return self.batchsz
        return len(self.train_x_batch)
