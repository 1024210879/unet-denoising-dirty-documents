from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import transforms as Transforms


class UNetDataset(Dataset):
    def __init__(self, dir_train, dir_mask, transform=None):
        self.dirTrain = dir_train
        self.dirMask = dir_mask
        self.transform = transform
        self.dataTrain = [os.path.join(self.dirTrain, filename)
                          for filename in os.listdir(self.dirTrain)
                          if filename.endswith('.jpg') or filename.endswith('.png')]
        self.dataMask = [os.path.join(self.dirMask, filename)
                         for filename in os.listdir(self.dirMask)
                         if filename.endswith('.jpg') or filename.endswith('.png')]
        self.trainDataSize = len(self.dataTrain)
        self.maskDataSize = len(self.dataMask)

    def __getitem__(self, index):
        assert self.trainDataSize == self.maskDataSize
        image = cv2.imread(self.dataTrain[index])
        label = cv2.imread(self.dataMask[index])

        if self.transform:
            for method in self.transform:
                image, label = method(image, label)

        return image[np.newaxis], label[np.newaxis]

    def __len__(self):
        assert self.trainDataSize == self.maskDataSize
        return self.trainDataSize


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    transforms = [
        Transforms.ToGray(),
        Transforms.RandomCrop(48, 48)
    ]
    dataset = UNetDataset('./data/train', './data/train_cleaned', transform=transforms)
    dataLoader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

    for index, (batch_x, batch_y) in enumerate(dataLoader):
        print(batch_x.size(), batch_y.size())

        dis = batch_y[0][0].numpy()
        cv2.imshow("dis", dis)
        cv2.waitKey()
