from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data_path):
        train_data = np.load(os.path.join(data_path, 'train_data.npy'))
        train_label = np.load(os.path.join(data_path, 'train_label.npy'))
        train_idxs = np.load(os.path.join(data_path, 'train_idxs.npy'))
        self.data = list(zip(train_idxs, train_data, train_label))

    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = MyDataset('data')
    dataloader = DataLoader(dataset=dataset, batch_size=5, shuffle=False)
    for batch_data in dataloader:
        # print(count)
        # count += 1
        idxs, inputs, labels = batch_data
        for idx in idxs:
            print(idx.item())
        break