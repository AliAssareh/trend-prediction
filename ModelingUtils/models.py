from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from enum import Enum


class Decoder(Enum):
    decoder74 = 74
    decoder654 = 654
    decoder620 = 620
    decoder594 = 594


class NetWork(Enum):
    network74 = 74
    network654 = 654
    network620 = 620
    network594 = 594


class CandlesDataset(Dataset):
    def __init__(self, features_df, labels_df):
        assert len(features_df) == len(labels_df)
        self.features = features_df
        self.targets = labels_df

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.features[idx, :]
        y = self.targets[idx, :]

        return (x, y)


class ConvCandlesDataset(Dataset):
    def __init__(self, features_df, labels_df):
        try:
            assert len(features_df) == len(labels_df)
        except Exception as ve:
            raise Exception(f'length of features_df {len(features_df)} dose not math length '
                            f'of labels_df {len(labels_df)}')
        self.features = features_df
        self.targets = labels_df

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.features[idx, :].reshape(1, *np.shape(self.features[idx, :]))
        y = self.targets[idx, :]

        return (x, y)


class MixedConvNet(nn.Module):
    def __init__(self):
        super(MixedConvNet, self).__init__()
        self.btc = BTCConv()
        self.btcd = BTCDConv()
        self.total = TOTALConv()
        self.usdt = USDTDConv()
        self.dxy = DXYConv()
        self.gold = TOTALConv()
        self.spx = SPXConv()
        self.ukoil = UKOilConv()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0))
        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 80, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0))
        self.max2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1))
        self.batch2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.max3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1))
        self.batch3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv3d(1, 1, kernel_size=(128, 1, 1), stride=(1, 1, 1))
        self.batch4 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim():
        return 58

    def forward(self, x):
        x0 = self.btc(x[:, :, :, 0:74])
        x1 = self.btcd(x[:, :, :, 74:145])
        x2 = self.total(x[:, :, :, 145:217])
        x3 = self.usdt(x[:, :, :, 217:290])
        x4 = self.dxy(x[:, :, :, 290:344])
        x5 = self.gold(x[:, :, :, 344:416])
        x6 = self.spx(x[:, :, :, 416:483])
        x7 = self.ukoil(x[:, :, :, 483:589])
        x = torch.concat([x0, x1, x2, x3, x4, x5, x6, x7], dim=3)
        x = F.silu(self.conv1(x))
        x = self.max1(x)
        x = self.batch1(x)
        x = F.silu(self.conv2(x))
        x = self.max2(x)
        x = self.batch2(x)
        x = F.silu(self.conv3(x))
        x = self.max3(x)
        x = self.batch3(x)
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = F.silu(self.conv4(x))
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        x = self.batch4(x)
        x = x.view(x.size(0), x.size(2), x.size(3))
        return x


class MixedDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(MixedDecoder, self).__init__()
        self.l1 = nn.Linear(latent_dim, 7 * 8)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 7, 8))

        self.deconv1 = nn.ConvTranspose2d(1, 64, (3, 7), stride=(3, 7), padding=(0, 0), output_padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(64, 64, (3, 7), stride=(1, 1), padding=(1, 0), groups=2)
        self.batch2 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, 64, (1, 3), stride=(1, 1), padding=(0, 0), groups=2)
        self.batch3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 48, (3, 5), stride=(1, 1), padding=(1, 0), groups=2)
        self.batch4 = nn.BatchNorm2d(48)

        self.deconv5 = nn.ConvTranspose2d(48, 48, (1, 3), stride=(1, 3), padding=(0, 0), groups=2)
        self.batch5 = nn.BatchNorm2d(48)

        self.deconv6 = nn.ConvTranspose2d(48, 1, (3, 3), stride=(1, 1), padding=(1, 0), output_padding=(0, 0))

    def forward(self, x):
        x = F.silu(self.l1(x))
        x = self.unflatten(x)
        x = F.silu(self.deconv1(x))
        x = self.batch1(x)
        x = F.silu(self.deconv2(x))
        x = self.batch2(x)
        x = F.silu(self.deconv3(x))
        x = self.batch3(x)
        x = F.silu(self.deconv4(x))
        x = self.batch4(x)
        x = F.silu(self.deconv5(x))
        x = self.batch5(x)
        x = self.deconv6(x)
        return x


class AuxiliaryMixedConvNet(nn.Module):
    def __init__(self, net_code=0):
        super(AuxiliaryMixedConvNet, self).__init__()
        self.net_code = net_code
        self.btc = BTCConv()
        if net_code == 0:
            self.auxiliary = BTCConv()
        elif net_code == 1:
            self.auxiliary = BTCDConv()
        elif net_code == 2:
            self.auxiliary = TOTALConv()
        elif net_code == 3:
            self.auxiliary = USDTDConv()
        elif net_code == 4:
            self.auxiliary = DXYConv()
        elif net_code == 5:
            self.auxiliary = TOTALConv()  # The net_topology is the same for gold and total
        elif net_code == 6:
            self.auxiliary = SPXConv()
        elif net_code == 7:
            self.auxiliary = UKOilConv()
        else:
            raise Exception('This net_code is not defined!')

    def get_output_dim(self):
        return self.btc.get_output_dim() + self.auxiliary.get_output_dim()

    def forward(self, x):
        x0 = self.btc(x[:, :, :, 0:74])
        if self.net_code == 0:
            x1 = self.auxiliary(x[:, :, :, 0:74])
        elif self.net_code == 1:
            x1 = self.auxiliary(x[:, :, :, 74:145])
        elif self.net_code == 2:
            x1 = self.auxiliary(x[:, :, :, 145:217])
        elif self.net_code == 3:
            x1 = self.auxiliary(x[:, :, :, 217:290])
        elif self.net_code == 4:
            x1 = self.auxiliary(x[:, :, :, 290:344])
        elif self.net_code == 5:
            x1 = self.auxiliary(x[:, :, :, 344:416])
        elif self.net_code == 6:
            x1 = self.auxiliary(x[:, :, :, 416:483])
        elif self.net_code == 7:
            x1 = self.auxiliary(x[:, :, :, 483:589])
        else:
            raise Exception('This net_code is not defined!')
        x = torch.concat([x0, x1], dim=3)
        return x


class AuxiliaryMixedConvNet2(nn.Module):
    def __init__(self, net_code=0):
        super(AuxiliaryMixedConvNet2, self).__init__()
        self.net_code = net_code
        if net_code == 0:
            self.auxiliary = BTCConv()
        elif net_code == 1:
            self.auxiliary = BTCDConv()
        elif net_code == 2:
            self.auxiliary = TOTALConv()
        elif net_code == 3:
            self.auxiliary = USDTDConv()
        elif net_code == 4:
            self.auxiliary = DXYConv()
        elif net_code == 5:
            self.auxiliary = TOTALConv()  # The net_topology is the same for gold and total
        elif net_code == 6:
            self.auxiliary = SPXConv()
        elif net_code == 7:
            self.auxiliary = UKOilConv()
        else:
            raise Exception('This net_code is not defined!')

    def get_output_dim(self):
        return self.auxiliary.get_output_dim()

    def forward(self, x):
        if self.net_code == 0:
            x = self.auxiliary(x[:, :, :, 0:74])
        elif self.net_code == 1:
            x = self.auxiliary(x[:, :, :, 74:145])
        elif self.net_code == 2:
            x = self.auxiliary(x[:, :, :, 145:217])
        elif self.net_code == 3:
            x = self.auxiliary(x[:, :, :, 217:290])
        elif self.net_code == 4:
            x = self.auxiliary(x[:, :, :, 290:344])
        elif self.net_code == 5:
            x = self.auxiliary(x[:, :, :, 344:416])
        elif self.net_code == 6:
            x = self.auxiliary(x[:, :, :, 416:483])
        elif self.net_code == 7:
            x = self.auxiliary(x[:, :, :, 483:589])
        else:
            raise Exception('This net_code is not defined!')
        return x


class BTCConv(nn.Module):
    def __init__(self):
        super(BTCConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0))
        self.max2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1), padding=(0, 0))
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0))
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv3d(1, 1, kernel_size=(64, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.batch4 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim():
        return 12

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.max1(x)
        x = self.batch1(x)
        x = F.silu(self.conv2(x))
        x = self.max2(x)
        x = self.batch2(x)
        x = F.silu(self.conv3(x))
        x = self.batch3(x)
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = F.silu(self.conv4(x))
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        x = self.batch4(x)
        return x


class BTCDConv(nn.Module):
    def __init__(self):
        super(BTCDConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0))
        self.max2 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 1), padding=(0, 0))
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0))
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv3d(1, 1, kernel_size=(64, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.batch4 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim():
        return 9

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.max1(x)
        x = self.batch1(x)
        x = F.silu(self.conv2(x))
        x = self.max2(x)
        x = self.batch2(x)
        x = F.silu(self.conv3(x))
        x = self.batch3(x)
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = F.silu(self.conv4(x))
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        x = self.batch4(x)
        return x


class TOTALConv(nn.Module):
    def __init__(self):
        super(TOTALConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0))
        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0))
        self.max2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1), padding=(0, 0))
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0))
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv3d(1, 1, kernel_size=(64, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.batch4 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim():
        return 10

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.max1(x)
        x = self.batch1(x)
        x = F.silu(self.conv2(x))
        x = self.max2(x)
        x = self.batch2(x)
        x = F.silu(self.conv3(x))
        x = self.batch3(x)
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = F.silu(self.conv4(x))
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        x = self.batch4(x)
        return x


class USDTDConv(nn.Module):
    def __init__(self):
        super(USDTDConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0))
        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0))
        self.max2 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 1), padding=(0, 0))
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0))
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv3d(1, 1, kernel_size=(64, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.batch4 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim():
        return 9

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.max1(x)
        x = self.batch1(x)
        x = F.silu(self.conv2(x))
        x = self.max2(x)
        x = self.batch2(x)
        x = F.silu(self.conv3(x))
        x = self.batch3(x)
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = F.silu(self.conv4(x))
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        x = self.batch4(x)
        return x


class DXYConv(nn.Module):
    def __init__(self):
        super(DXYConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0))
        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.max2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1), padding=(0, 0))
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0))
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv3d(1, 1, kernel_size=(64, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.batch4 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim():
        return 8

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.max1(x)
        x = self.batch1(x)
        x = F.silu(self.conv2(x))
        x = self.max2(x)
        x = self.batch2(x)
        x = F.silu(self.conv3(x))
        x = self.batch3(x)
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = F.silu(self.conv4(x))
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        x = self.batch4(x)
        return x


class SPXConv(nn.Module):
    def __init__(self):
        super(SPXConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0))
        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0))
        self.max2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1), padding=(0, 0))
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0))
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv3d(1, 1, kernel_size=(64, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.batch4 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim():
        return 9

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.max1(x)
        x = self.batch1(x)
        x = F.silu(self.conv2(x))
        x = self.max2(x)
        x = self.batch2(x)
        x = F.silu(self.conv3(x))
        x = self.batch3(x)
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = F.silu(self.conv4(x))
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        x = self.batch4(x)
        return x


class UKOilConv(nn.Module):
    def __init__(self):
        super(UKOilConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0))
        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0))
        self.max2 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 1), padding=(0, 0))
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0))
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv3d(1, 1, kernel_size=(64, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.batch4 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim():
        return 9

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.max1(x)
        x = self.batch1(x)
        x = F.silu(self.conv2(x))
        x = self.max2(x)
        x = self.batch2(x)
        x = F.silu(self.conv3(x))
        x = self.batch3(x)
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = F.silu(self.conv4(x))
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        x = self.batch4(x)
        return x


class ConvNet74(nn.Module):
    def __init__(self):
        super(ConvNet74, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.max1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 0))
        self.max2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1), padding=(0, 0))
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 7), stride=(1, 1), padding=(1, 0))
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv3d(1, 1, kernel_size=(64, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.batch4 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim(input_dim):
        return input_dim - 2 - 48 - 4 - 2 - 6

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = self.max1(x)
        x = self.batch1(x)
        x = F.silu(self.conv2(x))
        x = self.max2(x)
        x = self.batch2(x)
        x = F.silu(self.conv3(x))
        x = self.batch3(x)
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = F.silu(self.conv4(x))
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        x = self.batch4(x)
        x = x.view(x.size(0), x.size(2), x.size(3))
        return x


class Decoder74(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder74, self).__init__()
        self.l1 = nn.Linear(latent_dim, 7 * 8)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 7, 8))

        self.deconv1 = nn.ConvTranspose2d(1, 64, (3, 5), stride=(3, 1), padding=(0, 0), output_padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(64, 64, (3, 7), stride=(1, 1), padding=(1, 0), groups=2)
        self.batch2 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, 64, (1, 3), stride=(1, 1), padding=(0, 0), groups=2)
        self.batch3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 48, (3, 5), stride=(1, 1), padding=(1, 0), groups=2)
        self.batch4 = nn.BatchNorm2d(48)

        self.deconv5 = nn.ConvTranspose2d(48, 48, (1, 3), stride=(1, 3), padding=(0, 0), groups=2)
        self.batch5 = nn.BatchNorm2d(48)

        self.deconv6 = nn.ConvTranspose2d(48, 1, (3, 3), stride=(1, 1), padding=(1, 0), output_padding=(0, 0))

    def forward(self, x):
        x = F.silu(self.l1(x))
        x = self.unflatten(x)
        x = F.silu(self.deconv1(x))
        x = self.batch1(x)
        x = F.silu(self.deconv2(x))
        x = self.batch2(x)
        x = F.silu(self.deconv3(x))
        x = self.batch3(x)
        x = F.silu(self.deconv4(x))
        x = self.batch4(x)
        x = F.silu(self.deconv5(x))
        x = self.batch5(x)
        x = self.deconv6(x)
        return x


class WideLSTMNet(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(WideLSTMNet, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_dim, 110, batch_first=True)
        self.fc = nn.Linear(110, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 110, dtype=torch.double).requires_grad_().to(self.device)
        c0 = torch.zeros(1, x.size(0), 110, dtype=torch.double).requires_grad_().to(self.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class DeepLSTMNet(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(DeepLSTMNet, self).__init__()
        self.device = device
        self.nn1 = 90
        self.nn2 = 66
        self.nn3 = 48
        self.nn4 = 24

        self.l1 = nn.LSTMCell(input_dim, self.nn1)
        self.l2 = nn.LSTMCell(self.nn1, self.nn2)
        self.l3 = nn.LSTMCell(self.nn2, self.nn3)
        self.l4 = nn.LSTMCell(self.nn3, self.nn4)
        self.fc = nn.Linear(self.nn4, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        n_steps = x.size(1)

        h1 = Variable(torch.zeros(batch_size, self.nn1, dtype=torch.double)).to(self.device).detach()
        c1 = Variable(torch.zeros(batch_size, self.nn1, dtype=torch.double)).to(self.device).detach()
        h2 = Variable(torch.zeros(batch_size, self.nn2, dtype=torch.double)).to(self.device).detach()
        c2 = Variable(torch.zeros(batch_size, self.nn2, dtype=torch.double)).to(self.device).detach()
        h3 = Variable(torch.zeros(batch_size, self.nn3, dtype=torch.double)).to(self.device).detach()
        c3 = Variable(torch.zeros(batch_size, self.nn3, dtype=torch.double)).to(self.device).detach()
        h4 = Variable(torch.zeros(batch_size, self.nn4, dtype=torch.double)).to(self.device).detach()
        c4 = Variable(torch.zeros(batch_size, self.nn4, dtype=torch.double)).to(self.device).detach()

        for i in range(n_steps):
            (h1, c1) = self.l1(x[:, i, :], (h1, c1))
            (h2, c2) = self.l2(h1, (h2, c2))
            (h3, c3) = self.l3(h2, (h3, c3))
            (h4, c4) = self.l4(h3, (h4, c4))

        return self.fc(h4)


class DeepLSTMNet2(nn.Module):
    def __init__(self, device):
        super(DeepLSTMNet2, self).__init__()
        self.device = device
        self.nn1 = 110
        self.nn2 = 80
        self.nn3 = 60
        self.nn4 = 30

        self.l1 = nn.LSTMCell(74, self.nn1)
        self.l2 = nn.LSTMCell(self.nn1, self.nn2)
        self.l3 = nn.LSTMCell(self.nn2, self.nn3)
        self.l4 = nn.LSTMCell(self.nn3, self.nn4)
        self.fc = nn.Linear(self.nn4, 1)

    def forward(self, x):
        batch_size = x.size(0)
        n_steps = x.size(1)

        h1 = Variable(torch.zeros(batch_size, self.nn1, dtype=torch.double)).to(self.device).detach()
        c1 = Variable(torch.zeros(batch_size, self.nn1, dtype=torch.double)).to(self.device).detach()
        h2 = Variable(torch.zeros(batch_size, self.nn2, dtype=torch.double)).to(self.device).detach()
        c2 = Variable(torch.zeros(batch_size, self.nn2, dtype=torch.double)).to(self.device).detach()
        h3 = Variable(torch.zeros(batch_size, self.nn3, dtype=torch.double)).to(self.device).detach()
        c3 = Variable(torch.zeros(batch_size, self.nn3, dtype=torch.double)).to(self.device).detach()
        h4 = Variable(torch.zeros(batch_size, self.nn4, dtype=torch.double)).to(self.device).detach()
        c4 = Variable(torch.zeros(batch_size, self.nn4, dtype=torch.double)).to(self.device).detach()

        for i in range(n_steps):
            (h1, c1) = self.l1(x[:, i, :], (h1, c1))
            (h2, c2) = self.l2(h1, (h2, c2))
            (h3, c3) = self.l3(h2, (h3, c3))
            (h4, c4) = self.l4(h3, (h4, c4))

        return self.fc(h4)


class WideDeepLSTMNet(nn.Module):
    def __init__(self, input_dim, device):
        super(WideDeepLSTMNet, self).__init__()
        self.deep_lstm = DeepLSTMNet(input_dim, 1, device)
        self.wide_lstm = WideLSTMNet(input_dim, 1, device)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x1, x2 = self.deep_lstm(x), self.wide_lstm(x)
        x = torch.cat((x1, x2), dim=1)
        x = F.silu(x)
        x = self.fc(x)
        return x


class ConvWideDeepLSTMNet(nn.Module):
    def __init__(self, input_dim, device):
        super(ConvWideDeepLSTMNet, self).__init__()
        self.conv_net = ConvNet74()
        lstm_input_dim = self.conv_net.get_output_dim(input_dim)
        self.deep_lstm = DeepLSTMNet(lstm_input_dim, 1, device)
        self.wide_lstm = WideLSTMNet(lstm_input_dim, 1, device)
        self.batch1 = nn.BatchNorm1d(2)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.conv_net(x)
        x1, x2 = self.deep_lstm(x), self.wide_lstm(x)
        x = torch.cat((x1, x2), dim=1)
        x = F.silu(x)
        x = self.batch1(x)
        x = self.fc(x)
        return x


class MixedConvWideDeepLSTMNet(nn.Module):
    def __init__(self, device):
        super(MixedConvWideDeepLSTMNet, self).__init__()
        self.conv_net = MixedConvNet()
        lstm_input_dim = self.conv_net.get_output_dim()
        self.deep_lstm = DeepLSTMNet(lstm_input_dim, 1, device)
        self.wide_lstm = WideLSTMNet(lstm_input_dim, 1, device)
        self.batch1 = nn.BatchNorm1d(2)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.conv_net(x)
        x1, x2 = self.deep_lstm(x), self.wide_lstm(x)
        x = torch.cat((x1, x2), dim=1)
        x = F.silu(x)
        x = self.batch1(x)
        x = self.fc(x)
        return x


class AuxiliaryMixedConvWideDeepLSTMNet(nn.Module):
    def __init__(self, net_code, device, aux_code=1):
        super(AuxiliaryMixedConvWideDeepLSTMNet, self).__init__()
        if aux_code == 1:
            self.conv_net = AuxiliaryMixedConvNet(net_code)
        else:
            self.conv_net = AuxiliaryMixedConvNet2(net_code)
        lstm_input_dim = self.conv_net.get_output_dim()
        self.deep_lstm = DeepLSTMNet(lstm_input_dim, 1, device)
        self.wide_lstm = WideLSTMNet(lstm_input_dim, 1, device)
        self.batch1 = nn.BatchNorm1d(2)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), x.size(2), x.size(3))
        x1, x2 = self.deep_lstm(x), self.wide_lstm(x)
        x = torch.cat((x1, x2), dim=1)
        x = F.silu(x)
        x = self.batch1(x)
        x = self.fc(x)
        return x


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, nca_dim, device, onca_c=20, kl_c=1):
        super(VariationalEncoder, self).__init__()
        self.conv_net = ConvNet74()
        self.check_nca_dim_be_smaller_than_latent_dim(latent_dim, nca_dim)
        lstm_input_dim = self.conv_net.get_output_dim(input_dim)
        self.deep_lstm = DeepLSTMNet(lstm_input_dim, 22, device)
        self.wide_lstm = WideLSTMNet(lstm_input_dim, 22, device)
        self.l1 = nn.Linear(44, 42)

        self.lm1 = nn.Linear(42, 40)
        self.ls1 = nn.Linear(42, 40)
        self.lm2 = nn.Linear(40, latent_dim)
        self.ls2 = nn.Linear(40, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

        self.device = device
        self.nca_dim = nca_dim
        self.onca_c = onca_c
        self.onca = 0
        self.kl_c = kl_c
        self.kl = 0

    def check_nca_dim_be_smaller_than_latent_dim(self, latent_dim, nca_dim):
        if nca_dim > latent_dim:
            raise Exception('latent_dim should be bigger than or equal to nca_dim')

    def kl_loss(self, mu, sigma):
        return self.kl_c * (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

    def o_nca_loss(self, x_encoded, labels):
        if labels == None:
            return 0
        x_croped = x_encoded[:, 0:self.nca_dim]
        x_reshaped = x_croped * torch.ones(x_croped.size(0), *x_croped.size(), requires_grad=True).to(self.device)
        new_labels = labels * torch.ones(labels.size(0), labels.size(0), requires_grad=True).to(self.device)

        probs = torch.exp(torch.neg(torch.sum(torch.pow(x_reshaped - torch.transpose(x_reshaped, 0, 1), 2), 2)))

        matches = torch.eq(new_labels, torch.transpose(new_labels, 0, 1))
        matchhed_probs = probs * matches

        sum_matched_probs = torch.sum(matchhed_probs, 0) - 1
        sum_total_probs = torch.sum(probs, 0) - 0.9999

        o_nca = torch.sum(sum_matched_probs / sum_total_probs) / x_croped.size(0)
        onca_loss_coef = 1 - o_nca
        return self.onca_c * onca_loss_coef

    def forward(self, x, y=None):
        x = self.conv_net(x)
        x1, x2 = self.deep_lstm(x), self.wide_lstm(x)
        x = torch.cat((x1, x2), dim=1)
        x = F.silu(self.l1(x))

        x_m, x_s = F.silu(self.lm1(x)), F.silu(self.ls1(x))
        mu, sigma = self.lm2(x_m), torch.exp(self.ls2(x_s))

        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = self.kl_loss(mu, sigma)
        self.onca = self.o_nca_loss(mu, y)
        return [mu, z]


class MixedVariationalEncoder(nn.Module):
    def __init__(self, latent_dim, nca_dim, device, onca_c=20, kl_c=1):
        super(MixedVariationalEncoder, self).__init__()
        self.conv_net = MixedConvNet()
        self.check_nca_dim_be_smaller_than_latent_dim(latent_dim, nca_dim)
        lstm_input_dim = self.conv_net.get_output_dim()
        self.deep_lstm = DeepLSTMNet(lstm_input_dim, 24, device)
        self.wide_lstm = WideLSTMNet(lstm_input_dim, 24, device)
        self.l1 = nn.Linear(48, 46)

        self.lm1 = nn.Linear(46, 44)
        self.ls1 = nn.Linear(42, 44)
        self.lm2 = nn.Linear(44, latent_dim)
        self.ls2 = nn.Linear(44, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

        self.device = device
        self.nca_dim = nca_dim
        self.onca_c = onca_c
        self.onca = 0
        self.kl_c = kl_c
        self.kl = 0

    def check_nca_dim_be_smaller_than_latent_dim(self, latent_dim, nca_dim):
        if nca_dim > latent_dim:
            raise Exception('latent_dim should be bigger than or equal to nca_dim')

    def kl_loss(self, mu, sigma):
        return self.kl_c * (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

    def o_nca_loss(self, x_encoded, labels):
        if labels == None:
            return 0
        x_croped = x_encoded[:, 0:self.nca_dim]
        x_reshaped = x_croped * torch.ones(x_croped.size(0), *x_croped.size(), requires_grad=True).to(self.device)
        new_labels = labels * torch.ones(labels.size(0), labels.size(0), requires_grad=True).to(self.device)

        probs = torch.exp(torch.neg(torch.sum(torch.pow(x_reshaped - torch.transpose(x_reshaped, 0, 1), 2), 2)))

        matches = torch.eq(new_labels, torch.transpose(new_labels, 0, 1))
        matchhed_probs = probs * matches

        sum_matched_probs = torch.sum(matchhed_probs, 0) - 1
        sum_total_probs = torch.sum(probs, 0) - 0.9999

        o_nca = torch.sum(sum_matched_probs / sum_total_probs) / x_croped.size(0)
        onca_loss_coef = 1 - o_nca
        return self.onca_c * onca_loss_coef

    def forward(self, x, y=None):
        x = self.conv_net(x)
        x1, x2 = self.deep_lstm(x), self.wide_lstm(x)
        x = torch.cat((x1, x2), dim=1)
        x = F.silu(self.l1(x))

        x_m, x_s = F.silu(self.lm1(x)), F.silu(self.ls1(x))
        mu, sigma = self.lm2(x_m), torch.exp(self.ls2(x_s))

        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = self.kl_loss(mu, sigma)
        self.onca = self.o_nca_loss(mu, y)
        return [mu, z]


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, nca_dim, device, onca_c, kl_c):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_dim, latent_dim, nca_dim, device, onca_c, kl_c)
        self.decoder = Decoder74(latent_dim)

    def forward(self, x, y, test=False):
        encoder_results = self.encoder(x, y)
        if test:
            encoded_x = encoder_results[0]
        else:
            encoded_x = encoder_results[1]
        return self.decoder(encoded_x)


class MixedVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim, nca_dim, device, onca_c, kl_c):
        super(MixedVariationalAutoencoder, self).__init__()
        self.encoder = MixedVariationalEncoder(latent_dim, nca_dim, device, onca_c, kl_c)
        self.decoder = MixedDecoder(latent_dim)

    def forward(self, x, y, test=False):
        encoder_results = self.encoder(x, y)
        if test:
            encoded_x = encoder_results[0]
        else:
            encoded_x = encoder_results[1]
        return self.decoder(encoded_x)


class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, latent_dim, nca_dim, device):
        super(FullyConnectedNet, self).__init__()
        self.encoder = VariationalEncoder(input_dim, latent_dim=latent_dim, nca_dim=nca_dim, device=device)
        self.nca_dim = nca_dim
        self.batch1 = nn.BatchNorm1d(nca_dim)
        self.layer1 = nn.Linear(nca_dim, 9)
        self.batch2 = nn.BatchNorm1d(9)
        self.layer2 = nn.Linear(9, 6)
        self.batch3 = nn.BatchNorm1d(6)
        self.layer3 = nn.Linear(6, 3)
        self.batch4 = nn.BatchNorm1d(3)
        self.layer4 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.encoder(x, None)[0]
        x = self.batch1(x[:, 0: self.nca_dim])
        x = F.silu(self.layer1(x))
        x = self.batch2(x)
        x = F.silu(self.layer2(x))
        x = self.batch3(x)
        x = F.silu(self.layer3(x))
        x = self.batch4(x)
        x = self.layer4(x)
        return x


def train_epoch(model, device, dataloader, loss_fn1, loss_fn2, w1=0.75, w2=0.25, optimizer=None):
    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(device, dtype=torch.double)
        y = y.to(device, dtype=torch.double)

        loss = w1 * loss_fn1(model(x), y) + w2 * loss_fn2(model(x), y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
    return total_loss / len(dataloader.dataset)


def test_epoch(model, device, dataloader, loss_fn1, loss_fn2, w1=0.75, w2=0.25):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, dtype=torch.double)
            y = y.to(device, dtype=torch.double)

            loss = w1 * loss_fn1(model(x), y) + w2 * loss_fn2(model(x), y)

            total_loss = total_loss + loss.item()

    return total_loss / len(dataloader.dataset)


def vae_train_epoch(model, device, dataloader, optimizer):
    model.train()
    recon_loss, onca_loss, kl_loss, total_loss = 0.0, 0.0, 0.0, 0.0

    for x, y in dataloader:
        x = x.to(device, dtype=torch.double)
        y = y.to(device, dtype=torch.double)

        x_hat = model(x, y)
        recon = ((x - x_hat) ** 2).sum()
        onca = model.encoder.onca
        kl = model.encoder.kl

        loss = recon * (onca + 1) + kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recon_loss = recon_loss + recon.item()
        onca_loss = onca_loss + onca.item()
        kl_loss = kl_loss + kl.item()
        total_loss = total_loss + loss.item()
    return total_loss / len(dataloader.dataset), recon_loss / len(dataloader.dataset), \
           onca_loss / len(dataloader.dataset), kl_loss / len(dataloader.dataset)


def vae_test_epoch(model, device, dataloader):
    model.eval()
    recon_loss, onca_loss, kl_loss, total_loss = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, dtype=torch.double)
            y = y.to(device, dtype=torch.double)

            recon = ((x - model(x, y, test=True)) ** 2).sum()
            onca = model.encoder.onca
            kl = model.encoder.kl

            loss = recon * (onca + 1) + kl

            recon_loss = recon_loss + recon.item()
            onca_loss = onca_loss + onca.item()
            kl_loss = kl_loss + kl.item()
            total_loss = total_loss + loss.item()

    return total_loss / len(dataloader.dataset), recon_loss / len(dataloader.dataset), \
           onca_loss / len(dataloader.dataset), kl_loss / len(dataloader.dataset)


def save_vae_on_validation_improvement(model, optimizer, epoch, name):
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'encoder_state_dict': model.encoder.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, f'torch_models//{name}.pth')


def vae_plot(model, device, dataset):
    encoded_samples = []
    for sample in tqdm(dataset):
        img = torch.tensor(sample[0].reshape(1, *np.shape(sample[0]))).to(device)
        label = sample[1]
        # Encode image
        model.eval()
        with torch.no_grad():
            encoded_img = model.encoder(img)[0]
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {"v1": encoded_img[0], 'v2': encoded_img[1], 'v3': encoded_img[2], 'label': label}
        encoded_samples.append(encoded_sample)

    encoded_samples = pd.DataFrame(encoded_samples)
    encoded_samples.head()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8))
    ax[0].scatter(encoded_samples.iloc[:, 0].values, encoded_samples.iloc[:, 1], s=3, c=encoded_samples.loc[:, 'label'],
                cmap='bwr')
    ax[0].set(xlabel="dim1", ylabel="dim2")
    im = ax[1].scatter(encoded_samples.iloc[:, 0].values, encoded_samples.iloc[:, 2], s=3,
                       c=encoded_samples.loc[:, 'label'], cmap='bwr')
    ax[1].set(xlabel="dim1", ylabel="dim3")
    fig.colorbar(im, ax=ax[1])
    fig.suptitle("three fist dimensions of the encoded data")
    plt.show()


def load_encoder(path, encoder):
    state = torch.load(path)
    encoder.load_state_dict(state['encoder_state_dict'])
    return encoder


def load_fcn(path, model):
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    return model


def save_model_on_validation_improvement(model, optimizer, epoch, name):
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, f'torch_models//{name}.pth')


def load_model(path, model, optimizer=None):
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    if optimizer == None:
        return model
    else:
        optimizer.load_state_dict(state['optimizer'])
        return model, optimizer


def predict(model, dataset, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataset):
            x = torch.tensor(sample[0].reshape(1, *np.shape(sample[0]))).to(device)
            y_hat = model(x).cpu().numpy()
            predictions.append(y_hat)
    return np.array(predictions)


def encode(model, dataset, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataset):
            x = torch.tensor(sample[0].reshape(1, *np.shape(sample[0]))).to(device)
            y_hat = model(x)[0].cpu().numpy()
            predictions.append(y_hat)
    return np.array(predictions)


def evaluate(y_test, preds, continuous=True, separator=None, multiclass=False):
    y_min = np.amin(y_test)
    y_max = np.amax(y_test)
    if continuous:
        r_squared_score = r2_score(y_test[:, 0], preds[:, 0])
        print('r^2 score: ', r_squared_score)
        x_min = np.amin(preds)
        x_max = np.amax(preds)
        _min = np.amin([y_min, x_min])
        _max = np.amax([y_max, x_max])
        plt.figure(figsize=(25, 4.8))
        plt.plot(preds[:, 0], y_test[:, 0], 'b*')
        plt.plot(preds[0:12, 0], y_test[0:12, 0], 'r*')
        plt.ylabel('label')
        plt.xlabel('prediction')
        plt.title('predictions vs labels')
        plt.plot([_min - 0.15, _max + 0.15], [_min - 0.15, _max + 0.15], 'k-')
        plt.vlines(np.average(y_test[:, 0]), ymin=y_min - 0.15, ymax=y_max + 0.15)
        plt.show()
    else:
        plt.figure(figsize=(25, 4.8))
        plt.plot(preds, y_test, 'b|')
        plt.plot(preds[0:12], y_test[0:12], 'r|')
        plt.ylabel('label')
        plt.xlabel('prediction')
        plt.title('predictions vs labels')
        plt.yticks([-1, 1])
        if separator is None:
            plt.vlines(np.average(y_test), ymin=y_min, ymax=y_max)
        else:
            plt.vlines(separator, ymin=y_min, ymax=y_max)
        plt.show()

        plt.figure(figsize=(6.4, 4.8))
        if multiclass:
            M1 = confusion_matrix(y_test.astype('int32'), preds.astype('int32'), normalize='true')
            disp1 = ConfusionMatrixDisplay(M1)
            disp1.plot()
            plt.title('confusion matrix1')
            plt.show()
            return None
        else:
            fpr, tpr, _ = roc_curve(y_test, preds)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.show()

            if separator is None:
                b_preds = preds > np.average(y_test)
                b_preds = b_preds.astype('int32')
            else:
                b_preds = preds > separator
                b_preds = b_preds.astype('int32')
            b_y_test = y_test > np.average(y_test)
            b_y_test = b_y_test.astype('int32')
            cm0 = confusion_matrix(b_y_test.astype('int32'), b_preds.astype('int32'), normalize=None)
            precision_r = cm0[1, 1] / (cm0[1, 1] + cm0[0, 1])  # tp/(tp+fp)
            precision_f = cm0[0, 0] / (cm0[0, 0] + cm0[1, 0])  # tn/(tn+fn)
            cm1 = confusion_matrix(b_y_test.astype('int32'), b_preds.astype('int32'), normalize='true')
            recall_r = cm1[1, 1]  # tp/(tp+fn)
            recall_f = cm1[0, 0]  # tn/(tn+fp)
            disp1 = ConfusionMatrixDisplay(cm1)
            fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
            disp1.plot(ax=ax)
            plt.title('confusion matrix')
            plt.show()

            accuracy = (cm0[0, 0] + cm0[1, 1]) / (cm0[0, 0] + cm0[0, 1] + cm0[1, 0] + cm0[1, 1])
            f_score_r = 2 * (precision_r * recall_r) / (precision_r + recall_r)
            f_score_f = 2 * (precision_f * recall_f) / (precision_f + recall_f)
            print(f'accuracy: {accuracy}')
            print(f'f_score_rise: {f_score_r}, precision_rise: {precision_r}, recall_rise: {recall_r}')
            print(f'f_score_fall: {f_score_f}, precision_fall: {precision_f}, recall_fall: {recall_f}')
            return roc_auc, accuracy


def plot_stats_box(y_test, preds, groups=30, add_plot=None):
    b_preds = preds > 0
    b_preds = b_preds.astype('int32')
    b_y_test = y_test > np.average(y_test)
    b_y_test = b_y_test.astype('int32')
    batch_size = int(len(b_preds) / groups)
    scores = []
    for i in range(groups):
        if i != groups-1:
            predictions = b_preds[i*batch_size: (i+1)*batch_size]
            labels = b_y_test[i*batch_size: (i+1)*batch_size]
        else:
            predictions = b_preds[i * batch_size:]
            labels = b_y_test[i * batch_size:]
        cm0 = confusion_matrix(labels, predictions, normalize=None)
        accuracy = (cm0[0, 0] + cm0[1, 1]) / (cm0[0, 0] + cm0[0, 1] + cm0[1, 0] + cm0[1, 1])
        scores.append(100 * accuracy)

    if add_plot is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8))
        ax.set_title(f'Boxplot of Accuracies on 128 hours basis')
        ax.boxplot(scores, notch=True)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7.5))
        ax.set_title('left: ' + add_plot.get('title_1', None) + '    right: ' + add_plot.get('title_2', None))
        ax.boxplot([scores, add_plot.get('scores')], notch=True)
    return scores


