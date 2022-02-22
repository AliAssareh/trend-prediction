from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from enum import Enum


class Decoder(Enum):
    decoder74 = 74
    decoder654 = 654
    decoder620 = 620
    decoder594 = 594


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


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=(3, 31), stride=(1, 1), padding=(1, 0))
        self.max1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.batch1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=(3, 31), stride=(1, 1), padding=(1, 0))
        self.conv3 = nn.Conv3d(1, 1, kernel_size=(48, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.batch2 = nn.BatchNorm2d(1)

    @staticmethod
    def get_output_dim(input_dim):
        return input_dim - 30 - 2 - 30

    def forward(self, x):
        # print('x_size0: ', x.size())
        x = F.silu(self.conv1(x))
        # print('x_size1: ', x.size())
        x = self.max1(x)
        x = self.batch1(x)
        # print('x_size2: ', x.size())
        x = F.silu(self.conv2(x))
        # print('x_size1: ', x.size())
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        # print('x_size2: ', x.size())
        x = self.conv3(x)
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))
        x = self.batch2(x)
        x = F.silu(x)
        # x = self.batch3(x)
        # print('x_size4: ', x.size())
        x = x.view(x.size(0), x.size(2), x.size(3))
        # print('x_size5: ', x.size())
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


class WideDeepLSTMNet(nn.Module):
    def __init__(self, input_dim, device):
        super(WideDeepLSTMNet, self).__init__()
        self.deep_lstm = DeepLSTMNet(input_dim, 1, device)
        self.wide_lstm = WideLSTMNet(input_dim, 1, device)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x1, x2 = self.deep_lstm(x), self.wide_lstm(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


class ConvWideDeepLSTMNet(nn.Module):
    def __init__(self, input_dim, device):
        super(ConvWideDeepLSTMNet, self).__init__()
        self.conv_net = ConvNet()
        lstm_input_dim = self.conv_net.get_output_dim(input_dim)
        self.deep_lstm = DeepLSTMNet(lstm_input_dim, 1, device)
        self.wide_lstm = WideLSTMNet(lstm_input_dim, 1, device)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.conv_net(x)
        x1, x2 = self.deep_lstm(x), self.wide_lstm(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, nca_dim, device, onca_c=20, kl_c=1):
        super(VariationalEncoder, self).__init__()
        self.conv_net = ConvNet()
        self.check_nca_dim_be_smaller_than_latent_dim(latent_dim, nca_dim)
        lstm_input_dim = self.conv_net.get_output_dim(input_dim)
        self.deep_lstm = DeepLSTMNet(lstm_input_dim, 22, device)
        self.wide_lstm = WideLSTMNet(lstm_input_dim, 22, device)
        self.l1 = nn.Linear(44, 40)
        self.fc1 = nn.Linear(40, latent_dim)
        self.fc2 = nn.Linear(40, latent_dim)

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

        probs = torch.exp(torch.neg(torch.sum(torch.square(x_reshaped - torch.transpose(x_reshaped, 0, 1)), 2)))

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
        x = self.l1(x)
        mu = self.fc1(x)
        sigma = torch.exp(self.fc2(x))

        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = self.kl_loss(mu, sigma)
        self.onca = self.o_nca_loss(mu, y)
        return [mu, z]


class Decoder74(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder74, self).__init__()
        self.l1 = nn.Linear(latent_dim, 7 * 8)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 7, 8))

        self.deconv1 = nn.ConvTranspose2d(1, 48, (3, 5), stride=(3, 1), padding=(0, 0), output_padding=(0, 0))
        self.batch1 = nn.BatchNorm2d(48)
        self.deconv2 = nn.ConvTranspose2d(48, 24, (3, 31), stride=(1, 1), padding=(1, 0), output_padding=(0, 0),
                                          groups=2)
        self.batch2 = nn.BatchNorm2d(24)
        self.deconv3 = nn.ConvTranspose2d(24, 24, (3, 3), stride=(1, 1), padding=(1, 0), output_padding=(0, 0),
                                          groups=2)
        self.batch3 = nn.BatchNorm2d(24)
        self.deconv4 = nn.ConvTranspose2d(24, 1, (3, 31), stride=(1, 1), padding=(1, 0), output_padding=(0, 0))

    def forward(self, x):
        x = F.silu(self.l1(x))
        x = self.unflatten(x)
        x = F.silu(self.deconv1(x))
        x = self.batch1(x)
        # print('out1: ', x.size())
        x = F.silu(self.deconv2(x))
        x = self.batch2(x)
        # print('out2: ', x.size())
        x = F.silu(self.deconv3(x))
        x = self.batch3(x)
        # print('out3: ', x.size())
        x = self.deconv4(x)
        # print('out4: ', x.size())
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, nca_dim, device, onca_c, kl_c, decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_dim, latent_dim, nca_dim, device, onca_c, kl_c)
        if Decoder[decoder].value == Decoder.decoder74.value:
            self.decoder = Decoder74(latent_dim)
        else:
            raise Exception('this decoder is not implemented yet!')

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
    state = {'epoch': epoch, 'vae_state_dict': model.state_dict(), 'encoder_state_dict': model.encoder.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, f'torch_models//{name}.pth')


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


def load_model(path, model, optimizer):
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
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
        plt.figure()
        plt.plot(preds[:, 0], y_test[:, 0], 'b*')
        plt.plot(preds[0:12, 0], y_test[0:12, 0], 'r*')
        plt.ylabel('label')
        plt.xlabel('prediction')
        plt.title('predictions vs labels')
        plt.plot([_min - 0.15, _max + 0.15], [_min - 0.15, _max + 0.15], 'k-')
        plt.vlines(np.average(y_test[:, 0]), ymin=y_min - 0.15, ymax=y_max + 0.15)
        plt.show()
    else:
        plt.figure()
        plt.plot(preds, y_test, 'b|')
        plt.plot(preds[0:12], y_test[0:12], 'r|')
        plt.ylabel('label')
        plt.xlabel('prediction')
        plt.title('predictions vs labels')
        if separator is None:
            plt.vlines(np.average(y_test), ymin=y_min, ymax=y_max)
        else:
            plt.vlines(separator, ymin=y_min, ymax=y_max)
        plt.show()

        plt.figure()
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
            plt.figure()
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
            M1 = confusion_matrix(b_y_test.astype('int32'), b_preds.astype('int32'), normalize='true')
            disp1 = ConfusionMatrixDisplay(M1)
            disp1.plot()
            plt.title('confusion matrix1')
            plt.show()

            accuracy = (y_test.shape[0] - np.sum(np.abs(b_y_test - b_preds))) / y_test.shape[0]
            print('accuracy: ', accuracy)
            f_score = 2 * (M1[0, 0] * M1[1, 1]) / (M1[0, 0] + M1[1, 1])
            print('F_score: ', f_score)
            return roc_auc, accuracy
