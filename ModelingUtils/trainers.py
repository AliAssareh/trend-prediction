import time

import numpy as np
import torch
from sklearn.svm import SVC
from torch import nn

from FeatureEngineeringUtils.btc_feature_engineering_utils import TrainTestValidationLoader, \
    MixedDataTrainTestValidationLoader
from ModelingUtils.models import VariationalAutoencoder, FullyConnectedNet, vae_train_epoch, vae_test_epoch, \
    VariationalEncoder, encode, ConvWideDeepLSTMNet, MixedConvWideDeepLSTMNet, AuxiliaryMixedConvWideDeepLSTMNet, \
    vae_plot
from ModelingUtils.models import save_model_on_validation_improvement, load_model, load_fcn, load_encoder, predict, \
    evaluate, plot_stats_box
from ModelingUtils.models import train_epoch, test_epoch, save_vae_on_validation_improvement, ConvCandlesDataset


class Cwdn:
    def __init__(self, model_name, target, suffix, batch_size, max_epochs=50):
        self.model_name = model_name
        self.target = target
        self.suffix = suffix
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.device = torch.device('cpu')
        print(self.device)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.input_dim = None

        self.train_loader = None
        self.test_loader = None

        self.cwdn = None

        self.generate_prerequisite()

    def generate_prerequisite(self):
        self.generate_input_dim_and_datasets()
        self.generate_data_loaders()

    def generate_input_dim_and_datasets(self):
        data_loader = TrainTestValidationLoader(self.model_name, target=self.target, training_portion=0.75,
                                                validation_portion=0.4, original_time_frame=240)

        train_features, train_labels = data_loader.get_reframed_train_data()
        self.input_dim = np.shape(train_features)[2]
        self.train_dataset = ConvCandlesDataset(train_features, train_labels.reshape(-1, 1))

        test_features, test_labels = data_loader.get_reframed_test_data()
        self.test_dataset = ConvCandlesDataset(test_features, test_labels.reshape(-1, 1))

        val_features, val_labels = data_loader.get_reframed_val_data()
        self.val_dataset = ConvCandlesDataset(val_features, val_labels.reshape(-1, 1))

    def generate_data_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def generate_model(self, seed):
        torch.manual_seed(seed)

        self.cwdn = ConvWideDeepLSTMNet(self.input_dim, device=self.device)

        model_data = self.cwdn.to(self.device, dtype=torch.double)
        print(model_data)

    def get_proper_network_for_given_model(self):
        if self.model_name in ['btc_4h_for_target1_1', 'btc_1h_for_target1_1', 'btc_15m_for_target1_1']:
            return 'network74'
        elif self.model_name == 'mix_4h_for_target1_1':
            return 'network620'
        elif self.model_name == 'mix_1h_for_target1_1':
            return 'network594'
        elif self.model_name == 'mix_15m_for_target1_1':
            return 'network654'
        else:
            raise Exception('Given model have no predefined decoder yet!')

    def train_model(self, lr=5e-4, weight_decay=2e-3, patience=3, just_load=False):
        if just_load is False:
            adam_optimizer2 = torch.optim.Adam(self.cwdn.parameters(), lr=lr, weight_decay=weight_decay)

            loss_fn1 = nn.MSELoss(reduction='mean')
            loss_fn2 = nn.L1Loss(reduction='mean')

            best_loss = np.inf
            patience = patience
            last_improvement = 0

            num_epochs = 50
            epoch = 1
            flag = True
            while flag is True:
                start_time = time.time()

                total_loss_train = train_epoch(self.cwdn, self.device, self.train_loader, loss_fn1, loss_fn2,
                                               optimizer=adam_optimizer2)
                total_loss_test = test_epoch(self.cwdn, self.device, self.test_loader, loss_fn1, loss_fn2)

                if epoch % 1 == 0:
                    print('\n EPOCH {}/{} \t total_train {:.7f} \t total_test {:.7f} \t execution time: {:.0f}'.format(
                        epoch, num_epochs, total_loss_train, total_loss_test, (time.time() - start_time)))

                if total_loss_test < best_loss:
                    save_model_on_validation_improvement(self.cwdn, adam_optimizer2, epoch,
                                                         f'{self.model_name}_cwdn_{self.suffix}')
                    best_loss = total_loss_test
                    last_improvement = epoch

                if epoch - last_improvement >= patience:
                    flag = False

                if epoch >= num_epochs:
                    flag = False
                epoch = epoch + 1

        self.cwdn = load_model(f'torch_models//{self.model_name}_cwdn_{self.suffix}.pth', self.cwdn)

    def generate_results(self):
        model = self.cwdn
        train_preds = predict(model, self.train_dataset, self.device)
        evaluate(self.train_dataset.targets, train_preds.reshape(-1, 1), continuous=False, separator=0)
        test_preds = predict(model, self.test_dataset, self.device)
        evaluate(self.test_dataset.targets, test_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)
        val_preds = predict(model, self.val_dataset, self.device)
        evaluate(self.val_dataset.targets, val_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)
        self.save_conv_net(model)

    def save_conv_net(self, model):
        cwdn = load_model('torch_models//cwdn2.pth', model)
        conv_net = cwdn.conv_net

        state = {'state_dict': conv_net.state_dict()}
        torch.save(state, f'torch_models//{self.model_name}_conv_net_{self.suffix}.pth')

    def generate_test_stats(self):
        model = self.cwdn
        val_preds = predict(model, self.val_dataset, self.device)
        scores = plot_stats_box(self.val_dataset.targets, val_preds.reshape(-1, 1))
        return scores

    def generate_reduced_val_results(self):
        model = self.cwdn
        val_preds = predict(model, self.val_dataset, self.device)
        labels = self.val_dataset.targets[-666:]
        print('number of samples in reduced validation set', len(labels))
        preds = val_preds[-666:]
        evaluate(labels, preds.reshape(-1, 1), continuous=False, separator=0)


class NVAE:
    def __init__(self, model_name, target, suffix, batch_size=127, latent_dim=30, nca_dim=20, onca_c=3, kl_c=1,
                 max_epochs=12):
        self.model_name = model_name
        self.target = target
        self.suffix = suffix
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.nca_dim = nca_dim
        self.onca_c = onca_c
        self.kl_c = kl_c
        self.max_epochs = max_epochs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.device)

        self.data_loader = None

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.input_dim = None

        self.train_loader = None
        self.test_loader = None

        self.vae = None
        self.fcn = None
        self.svm = None

        self.train_encoded = None
        self.test_encoded = None
        self.val_encoded = None

        self.generate_prerequisite()

    def generate_prerequisite(self):
        self.generate_input_dim_and_datasets()
        self.generate_data_loaders()

    def generate_input_dim_and_datasets(self):
        self.data_loader = TrainTestValidationLoader(self.model_name, target=self.target, training_portion=0.75,
                                                     validation_portion=0.4, original_time_frame=240)

        train_features, train_labels = self.data_loader.get_reframed_train_data()
        self.input_dim = np.shape(train_features)[2]
        self.train_dataset = ConvCandlesDataset(train_features, train_labels.reshape(-1, 1))

        test_features, test_labels = self.data_loader.get_reframed_test_data()
        self.test_dataset = ConvCandlesDataset(test_features, test_labels.reshape(-1, 1))

        val_features, val_labels = self.data_loader.get_reframed_val_data()
        self.val_dataset = ConvCandlesDataset(val_features, val_labels.reshape(-1, 1))

    def generate_data_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def generate_autoencoder(self, seed=0, silent=False):
        torch.manual_seed(seed)

        self.vae = VariationalAutoencoder(self.input_dim, latent_dim=self.latent_dim, nca_dim=self.nca_dim,
                                          device=self.device, onca_c=self.onca_c, kl_c=self.kl_c)

        self.vae.encoder.conv_net = load_model(f'torch_models//{self.model_name}_conv_net_{self.suffix}.pth',
                                               self.vae.encoder.conv_net)

        for name, child in self.vae.named_children():
            if name == 'encoder':
                for name2, child2 in child.named_children():
                    if name2 == 'conv_net':
                        for param in child2.parameters():
                            param.requires_grad = False
                        if silent:
                            pass
                        else:
                            print('conv_net weights frozen')

        model_data = self.vae.to(self.device, dtype=torch.double)
        if silent:
            pass
        else:
            print(model_data)

    def train_auto_encoder(self, lr=2e-3, weight_decay=5e-3, patience=3, just_load=False):
        if just_load is False:
            params = filter(lambda p: p.requires_grad, self.vae.parameters())
            adam_optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

            best_loss = np.inf
            patience = patience
            last_improvement = 0

            num_epochs = self.max_epochs
            epoch = 1
            flag = True
            while flag is True:
                start_time = time.time()

                total_loss, recon_loss, onca, kl = vae_train_epoch(self.vae, self.device, self.train_loader,
                                                                   adam_optimizer)
                total_loss2, recon_loss2, onca2, kl2 = vae_test_epoch(self.vae, self.device, self.test_loader)

                if epoch % 1 == 0:
                    print(
                        '\n EPOCH {}/{} \t total_train {:.3f} \t reconstruction loss {:.3f} \t onca {:.5f} \t kl {:.3f}'
                        ''.format(epoch, num_epochs, total_loss, recon_loss, onca, kl))
                    print('\t\t total_test {:.3f} \t reconstruction loss {:.3f} \t onca {:.5f} \t kl {:.3f} \t '
                          'execution time: {:.0f}s'.format(total_loss2, recon_loss2, onca2, kl2,
                                                           time.time() - start_time))

                # if epoch % 2 == 0:
                #     vae_test_plot(self.vae, self.device, self.test_dataset)

                if total_loss2 < best_loss:
                    save_vae_on_validation_improvement(self.vae, adam_optimizer, epoch,
                                                       f'{self.model_name}_vae_{self.suffix}')
                    best_loss = total_loss2
                    last_improvement = epoch

                if epoch - last_improvement >= patience:
                    flag = False

                if epoch >= num_epochs:
                    flag = False
                epoch = epoch + 1

        self.vae = load_model(f'torch_models//{self.model_name}_vae_{self.suffix}.pth', self.vae)

    def plot_encoded_space(self):
        vae_plot(self.vae, self.device, self.test_dataset)

    def generate_fully_connected_network(self, seed=0):
        torch.manual_seed(seed)

        self.fcn = FullyConnectedNet(self.input_dim, latent_dim=self.latent_dim, nca_dim=self.nca_dim,
                                     device=self.device)

        self.fcn.encoder.load_state_dict(self.vae.encoder.state_dict())

        for name, child in self.fcn.named_children():
            if name == 'encoder':
                for param in child.parameters():
                    param.requires_grad = False

        _ = self.fcn.to(self.device, dtype=torch.double)

    def train_fully_connected_net(self, lr=5e-4, weight_decay=2e-3, w1=0.75, w2=0.25, patience=3, just_load=False):
        if just_load is False:
            params = filter(lambda p: p.requires_grad, self.fcn.parameters())
            adam_optimizer2 = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

            loss_fn1 = nn.MSELoss(reduction='mean')
            loss_fn2 = nn.L1Loss(reduction='mean')

            best_loss = np.inf
            patience = patience
            last_improvement = 0

            num_epochs = 50
            epoch = 1
            flag = True
            while flag is True:
                start_time = time.time()

                total_loss_train = train_epoch(self.fcn, self.device, self.train_loader, loss_fn1, loss_fn2,
                                               w1=w1, w2=w2, optimizer=adam_optimizer2)
                total_loss_test = test_epoch(self.fcn, self.device, self.test_loader, loss_fn1, loss_fn2)

                if epoch % 1 == 0:
                    print('\n EPOCH {}/{} \t total_train {:.7f} \t total_test {:.7f} \t execution time: {:.0f}'.format(
                        epoch, num_epochs, total_loss_train, total_loss_test, (time.time() - start_time)))

                if total_loss_test < best_loss:
                    save_model_on_validation_improvement(self.fcn, adam_optimizer2, epoch,
                                                         f'{self.model_name}_fcn_{self.suffix}')
                    best_loss = total_loss_test
                    last_improvement = epoch

                if epoch - last_improvement >= patience:
                    flag = False

                if epoch >= num_epochs:
                    flag = False
                epoch = epoch + 1

        self.fcn = load_fcn(f'torch_models//{self.model_name}_fcn_{self.suffix}.pth', self.fcn)

    def generate_predictions(self):
        model = self.fcn
        train_preds = predict(model, self.train_dataset, self.device)
        test_preds = predict(model, self.test_dataset, self.device)
        val_preds = predict(model, self.val_dataset, self.device)
        return train_preds, test_preds, val_preds

    def generate_fc_results(self):
        model = self.fcn
        train_preds = predict(model, self.train_dataset, self.device)
        evaluate(self.train_dataset.targets, train_preds.reshape(-1, 1), continuous=False, separator=0)
        test_preds = predict(model, self.test_dataset, self.device)
        evaluate(self.test_dataset.targets, test_preds.reshape(-1, 1), continuous=False, separator=0)
        val_preds = predict(model, self.val_dataset, self.device)
        evaluate(self.val_dataset.targets, val_preds.reshape(-1, 1), continuous=False, separator=0)

    def generate_reduced_val_results(self):
        model = self.fcn
        val_preds = predict(model, self.val_dataset, self.device)
        labels = self.val_dataset.targets[-666:]
        print('number of samples in reduced validation set', len(labels))
        preds = val_preds[-666:]
        evaluate(labels, preds.reshape(-1, 1), continuous=False, separator=0)

    def generate_fc_test_stats(self, add_plot=None):
        model = self.fcn
        val_preds = predict(model, self.val_dataset, self.device)
        scores = plot_stats_box(self.val_dataset.targets, val_preds.reshape(-1, 1), add_plot=add_plot)

    def generate_svm(self, kernel='rbf', degree=3, r=0, c=1.0, gamma='scale'):
        self.svm = SVC(kernel=kernel, degree=degree, coef0=r, C=c, gamma=gamma)
        self.encoder = self.vae.encoder
        self.encoder.to(self.device, dtype=torch.double)

    def train_svm(self):
        if self.train_encoded is None:
            train_preds = encode(self.encoder, self.train_dataset, self.device)
            x_train = train_preds[:, :, 0:self.nca_dim]
            print(np.shape(x_train))
            self.train_encoded = x_train.reshape(np.shape(x_train)[0], np.shape(x_train)[2])

        y_train = self.train_dataset.targets.reshape(-1)
        self.svm.fit(self.train_encoded, y_train)

    def generate_svm_results(self):
        if self.train_encoded is None:
            train_preds = encode(self.encoder, self.train_dataset, self.device)
            x_train = train_preds[:, :, 0:self.nca_dim]
            self.train_encoded = x_train.reshape(np.shape(x_train)[0], np.shape(x_train)[2])

        train_preds = self.svm.predict(self.train_encoded)
        evaluate(self.train_dataset.targets, train_preds.reshape(-1, 1), continuous=False, separator=0,
                 multiclass=False)

        if self.test_encoded is None:
            test_preds = encode(self.encoder, self.test_dataset, self.device)
            x_test = test_preds[:, :, 0:self.nca_dim]
            self.test_encoded = x_test.reshape(np.shape(x_test)[0], np.shape(x_test)[2])
        test_preds = self.svm.predict(self.test_encoded)
        evaluate(self.test_dataset.targets, test_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)

        if self.val_encoded is None:
            val_preds = encode(self.encoder, self.val_dataset, self.device)
            x_val = val_preds[:, :, 0:self.nca_dim]
            self.val_encoded = x_val.reshape(np.shape(x_val)[0], np.shape(x_val)[2])
        val_preds = self.svm.predict(self.val_encoded)
        evaluate(self.val_dataset.targets, val_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)

    def generate_svm_test_stats(self):
        if self.val_encoded is None:
            val_preds = encode(self.encoder, self.val_dataset, self.device)
            x_val = val_preds[:, 0:self.nca_dim]
            self.val_encoded = x_val.reshape(np.shape(x_val)[0], np.shape(x_val)[2])
        val_preds = self.svm.predict(self.val_encoded)
        plot_stats_box(self.val_dataset.targets, val_preds.reshape(-1, 1))


class AuxiliaryNet:
    def __init__(self, model_name, target, net_code, suffix, batch_size, max_epochs=50):
        self.model_name = model_name
        self.target = target
        self.net_code = net_code
        self.suffix = suffix
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.device)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.input_dim = None

        self.train_loader = None
        self.test_loader = None

        self.cwdn = None

        self.generate_prerequisite()

    def generate_prerequisite(self):
        self.generate_input_dim_and_datasets()
        self.generate_data_loaders()

    def generate_input_dim_and_datasets(self):
        data_loader = MixedDataTrainTestValidationLoader(self.model_name, target=self.target, training_portion=0.6385,
                                                         validation_portion=0.4, original_time_frame=240)

        train_features, train_labels = data_loader.get_reframed_train_data()
        # print(np.shape(train_labels))
        self.train_dataset = ConvCandlesDataset(train_features, train_labels.reshape(-1, 1))

        test_features, test_labels = data_loader.get_reframed_test_data()
        # print(np.shape(test_labels))
        self.test_dataset = ConvCandlesDataset(test_features, test_labels.reshape(-1, 1))

        val_features, val_labels = data_loader.get_reframed_val_data()
        # print(np.shape(val_labels))
        self.val_dataset = ConvCandlesDataset(val_features, val_labels.reshape(-1, 1))

    def generate_data_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def generate_model(self, seed, aux_code=1):
        torch.manual_seed(seed)

        self.cwdn = AuxiliaryMixedConvWideDeepLSTMNet(self.net_code, device=self.device, aux_code=aux_code)

        if aux_code == 1:
            self.cwdn.conv_net.btc = load_model(f'torch_models//btc_4h_for_target1_1_conv_net_1.pth',
                                                self.cwdn.conv_net.btc)

            for name, child in self.cwdn.named_children():
                if name == 'conv_net':
                    for name2, child2 in child.named_children():
                        if name2 == 'btc':
                            for param in child2.parameters():
                                param.requires_grad = False
                            print('btc_conv_net weights frozen')

        model_data = self.cwdn.to(self.device, dtype=torch.double)
        print(model_data)

    def train_model(self, lr=5e-4, weight_decay=2e-3, w1=0.75, w2=0.25, patience=3, just_load=False):
        if just_load is False:
            adam_optimizer2 = torch.optim.Adam(self.cwdn.parameters(), lr=lr, weight_decay=weight_decay)

            loss_fn1 = nn.MSELoss(reduction='mean')
            loss_fn2 = nn.L1Loss(reduction='mean')

            best_loss = np.inf
            patience = patience
            last_improvement = 0

            num_epochs = 50
            epoch = 1
            flag = True
            while flag is True:
                start_time = time.time()

                total_loss_train = train_epoch(self.cwdn, self.device, self.train_loader, loss_fn1, loss_fn2,
                                               w1=w1, w2=w2, optimizer=adam_optimizer2)
                total_loss_test = test_epoch(self.cwdn, self.device, self.test_loader, loss_fn1, loss_fn2, w1, w2)

                if epoch % 1 == 0:
                    print('\n EPOCH {}/{} \t total_train {:.7f} \t total_test {:.7f} \t execution time: {:.0f}'.format(
                        epoch, num_epochs, total_loss_train, total_loss_test, (time.time() - start_time)))

                if total_loss_test < best_loss:
                    save_model_on_validation_improvement(self.cwdn, adam_optimizer2, epoch,
                                                         f'{self.model_name}_ax_mixcwdn_{self.net_code}_{self.suffix}')
                    best_loss = total_loss_test
                    last_improvement = epoch

                if epoch - last_improvement >= patience:
                    flag = False

                if epoch >= num_epochs:
                    flag = False
                epoch = epoch + 1

        self.cwdn = load_model(f'torch_models//{self.model_name}_ax_mixcwdn_{self.net_code}_{self.suffix}.pth',
                               self.cwdn)

    def generate_results(self):
        model = self.cwdn
        train_preds = predict(model, self.train_dataset, self.device)
        evaluate(self.train_dataset.targets, train_preds.reshape(-1, 1), continuous=False, separator=0)
        test_preds = predict(model, self.test_dataset, self.device)
        evaluate(self.test_dataset.targets, test_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)
        val_preds = predict(model, self.val_dataset, self.device)
        evaluate(self.val_dataset.targets, val_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)

    def save_conv_net(self):
        conv_net = self.cwdn.conv_net.auxiliary

        state = {'state_dict': conv_net.state_dict()}
        torch.save(state, f'torch_models//{self.model_name}_ax_conv_net_{self.net_code}_{self.suffix}.pth')

    def generate_test_stats(self):
        model = self.cwdn
        val_preds = predict(model, self.val_dataset, self.device)
        scores = plot_stats_box(self.val_dataset.targets, val_preds.reshape(-1, 1))
        return scores


class MixCwdn:
    def __init__(self, model_name, target, suffix, batch_size, max_epochs=50):
        self.model_name = model_name
        self.target = target
        self.suffix = suffix
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.device = torch.device('cpu')
        print(self.device)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.input_dim = None

        self.train_loader = None
        self.test_loader = None

        self.cwdn = None

        self.generate_prerequisite()

    def generate_prerequisite(self):
        self.generate_input_dim_and_datasets()
        self.generate_data_loaders()

    def generate_input_dim_and_datasets(self):
        data_loader = MixedDataTrainTestValidationLoader(self.model_name, target=self.target, training_portion=0.6385,
                                                         validation_portion=0.4, original_time_frame=240)

        train_features, train_labels = data_loader.get_reframed_train_data()
        self.train_dataset = ConvCandlesDataset(train_features, train_labels.reshape(-1, 1))

        test_features, test_labels = data_loader.get_reframed_test_data()
        self.test_dataset = ConvCandlesDataset(test_features, test_labels.reshape(-1, 1))

        val_features, val_labels = data_loader.get_reframed_val_data()
        self.val_dataset = ConvCandlesDataset(val_features, val_labels.reshape(-1, 1))

    def generate_data_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def generate_model(self, seed):
        torch.manual_seed(seed)

        self.cwdn = MixedConvWideDeepLSTMNet(device=self.device)

        self.cwdn.conv_net.btc = load_model(f'torch_models//btc_4h_for_target1_1_conv_net_1.pth',
                                            self.cwdn.conv_net.btc)
        self.cwdn.conv_net.btcd = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_1_{self.suffix}.pth',
                                             self.cwdn.conv_net.btcd)
        self.cwdn.conv_net.total = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_2_{self.suffix}.pth',
                                              self.cwdn.conv_net.total)
        self.cwdn.conv_net.usdt = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_3_{self.suffix}.pth',
                                             self.cwdn.conv_net.usdt)
        self.cwdn.conv_net.dxy = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_4_{self.suffix}.pth',
                                            self.cwdn.conv_net.dxy)
        self.cwdn.conv_net.gold = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_5_{self.suffix}.pth',
                                             self.cwdn.conv_net.gold)
        self.cwdn.conv_net.spx = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_6_{self.suffix}.pth',
                                            self.cwdn.conv_net.spx)
        self.cwdn.conv_net.ukoil = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_7_{self.suffix}.pth',
                                              self.cwdn.conv_net.ukoil)

        for name, child in self.cwdn.named_children():
            if name == 'conv_net':
                for name2, child2 in child.named_children():
                    if name2 in ['btc', 'btcd', 'total', 'usdt', 'dxy', 'gold', 'spx', 'ukoil']:
                        for param in child2.parameters():
                            param.requires_grad = False
                        print(f'{name2}_conv_net weights frozen')

        model_data = self.cwdn.to(self.device, dtype=torch.double)
        print(model_data)

    def generate_model2(self):
        self.cwdn = MixedConvWideDeepLSTMNet(device=self.device)

        self.cwdn = load_model(f'torch_models//{self.model_name}_mixcwdn_{self.suffix}.pth', self.cwdn)

        self.cwdn.conv_net.btc = load_model(f'torch_models//btc_4h_for_target1_1_conv_net_1.pth',
                                            self.cwdn.conv_net.btc)
        self.cwdn.conv_net.btcd = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_1_{self.suffix}.pth',
                                             self.cwdn.conv_net.btcd)
        self.cwdn.conv_net.total = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_2_{self.suffix}.pth',
                                              self.cwdn.conv_net.total)
        self.cwdn.conv_net.usdt = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_3_{self.suffix}.pth',
                                             self.cwdn.conv_net.usdt)
        self.cwdn.conv_net.dxy = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_4_{self.suffix}.pth',
                                            self.cwdn.conv_net.dxy)
        self.cwdn.conv_net.gold = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_5_{self.suffix}.pth',
                                             self.cwdn.conv_net.gold)
        self.cwdn.conv_net.spx = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_6_{self.suffix}.pth',
                                            self.cwdn.conv_net.spx)
        self.cwdn.conv_net.ukoil = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_7_{self.suffix}.pth',
                                              self.cwdn.conv_net.ukoil)

        model_data = self.cwdn.to(self.device, dtype=torch.double)

    def train_model(self, lr=5e-4, weight_decay=2e-3, patience=3, w1=0.75, w2=0.25, just_load=False):
        if just_load is False:
            adam_optimizer2 = torch.optim.Adam(self.cwdn.parameters(), lr=lr, weight_decay=weight_decay)

            loss_fn1 = nn.MSELoss(reduction='mean')
            loss_fn2 = nn.L1Loss(reduction='mean')

            best_loss = np.inf
            patience = patience
            last_improvement = 0

            num_epochs = 50
            epoch = 1
            flag = True
            while flag is True:
                start_time = time.time()

                total_loss_train = train_epoch(self.cwdn, self.device, self.train_loader, loss_fn1, loss_fn2,
                                               w1=w1, w2=w2, optimizer=adam_optimizer2)
                total_loss_test = test_epoch(self.cwdn, self.device, self.test_loader, loss_fn1, loss_fn2)

                if epoch % 1 == 0:
                    print('\n EPOCH {}/{} \t total_train {:.7f} \t total_test {:.7f} \t execution time: {:.0f}'.format(
                        epoch, num_epochs, total_loss_train, total_loss_test, (time.time() - start_time)))

                if total_loss_test < best_loss:
                    save_model_on_validation_improvement(self.cwdn, adam_optimizer2, epoch,
                                                         f'{self.model_name}_mixcwdn_{self.suffix}')
                    best_loss = total_loss_test
                    last_improvement = epoch

                if epoch - last_improvement >= patience:
                    flag = False

                if epoch >= num_epochs:
                    flag = False
                epoch = epoch + 1

        self.cwdn = load_model(f'torch_models//{self.model_name}_mixcwdn_{self.suffix}.pth', self.cwdn)

    def generate_results(self):
        model = self.cwdn
        train_preds = predict(model, self.train_dataset, self.device)
        evaluate(self.train_dataset.targets, train_preds.reshape(-1, 1), continuous=False, separator=0)
        test_preds = predict(model, self.test_dataset, self.device)
        evaluate(self.test_dataset.targets, test_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)
        val_preds = predict(model, self.val_dataset, self.device)
        evaluate(self.val_dataset.targets, val_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)

    def generate_test_stats(self):
        model = self.cwdn
        val_preds = predict(model, self.val_dataset, self.device)
        scores = plot_stats_box(self.val_dataset.targets, val_preds.reshape(-1, 1), groups=20)
        return scores


class MixNVAE:
    def __init__(self, model_name, target, suffix, batch_size, max_epochs=50):
        self.model_name = model_name
        self.target = target
        self.suffix = suffix
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.device = torch.device('cpu')
        print(self.device)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.input_dim = None

        self.train_loader = None
        self.test_loader = None

        self.cwdn = None

        self.generate_prerequisite()

    def generate_prerequisite(self):
        self.generate_input_dim_and_datasets()
        self.generate_data_loaders()

    def generate_input_dim_and_datasets(self):
        data_loader = MixedDataTrainTestValidationLoader(self.model_name, target=self.target, training_portion=0.75,
                                                         validation_portion=0.4, original_time_frame=240)

        train_features, train_labels = data_loader.get_reframed_train_data()
        self.train_dataset = ConvCandlesDataset(train_features, train_labels.reshape(-1, 1))

        test_features, test_labels = data_loader.get_reframed_test_data()
        self.test_dataset = ConvCandlesDataset(test_features, test_labels.reshape(-1, 1))

        val_features, val_labels = data_loader.get_reframed_val_data()
        self.val_dataset = ConvCandlesDataset(val_features, val_labels.reshape(-1, 1))

    def generate_data_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def generate_model(self, seed):
        torch.manual_seed(seed)

        self.cwdn = MixedConvWideDeepLSTMNet(device=self.device)

        self.cwdn.conv_net.btc = load_model(f'torch_models//btc_4h_for_target1_1_conv_net_1.pth',
                                            self.cwdn.conv_net.btc)
        self.cwdn.conv_net.btcd = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_1_{self.suffix}.pth',
                                             self.cwdn.conv_net.btcd)
        self.cwdn.conv_net.total = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_2_{self.suffix}.pth',
                                              self.cwdn.conv_net.total)
        self.cwdn.conv_net.usdt = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_3_{self.suffix}.pth',
                                             self.cwdn.conv_net.usdt)
        self.cwdn.conv_net.dxy = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_4_{self.suffix}.pth',
                                            self.cwdn.conv_net.dxy)
        self.cwdn.conv_net.gold = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_5_{self.suffix}.pth',
                                             self.cwdn.conv_net.gold)
        self.cwdn.conv_net.spx = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_6_{self.suffix}.pth',
                                            self.cwdn.conv_net.spx)
        self.cwdn.conv_net.ukoil = load_model(f'torch_models//mix_4h_for_target1_1_ax_conv_net_7_{self.suffix}.pth',
                                              self.cwdn.conv_net.ukoil)

        for name, child in self.cwdn.named_children():
            if name == 'conv_net':
                for name2, child2 in child.named_children():
                    if name2 in ['btc', 'btcd', 'total', 'usdt', 'dxy', 'gold', 'spx', 'ukoil']:
                        for param in child2.parameters():
                            param.requires_grad = False
                        print(f'{name2}_conv_net weights frozen')

        model_data = self.cwdn.to(self.device, dtype=torch.double)
        print(model_data)

    def train_model(self, lr=5e-4, weight_decay=2e-3, patience=3, w1=0.75, w2=0.25, just_load=False):
        if just_load is False:
            adam_optimizer2 = torch.optim.Adam(self.cwdn.parameters(), lr=lr, weight_decay=weight_decay)

            loss_fn1 = nn.MSELoss(reduction='mean')
            loss_fn2 = nn.L1Loss(reduction='mean')

            best_loss = np.inf
            patience = patience
            last_improvement = 0

            num_epochs = 50
            epoch = 1
            flag = True
            while flag is True:
                start_time = time.time()

                total_loss_train = train_epoch(self.cwdn, self.device, self.train_loader, loss_fn1, loss_fn2,
                                               w1=w1, w2=w2, optimizer=adam_optimizer2)
                total_loss_test = test_epoch(self.cwdn, self.device, self.test_loader, loss_fn1, loss_fn2)

                if epoch % 1 == 0:
                    print('\n EPOCH {}/{} \t total_train {:.7f} \t total_test {:.7f} \t execution time: {:.0f}'.format(
                        epoch, num_epochs, total_loss_train, total_loss_test, (time.time() - start_time)))

                if total_loss_test < best_loss:
                    save_model_on_validation_improvement(self.cwdn, adam_optimizer2, epoch,
                                                         f'{self.model_name}_mixcwdn_{self.suffix}')
                    best_loss = total_loss_test
                    last_improvement = epoch

                if epoch - last_improvement >= patience:
                    flag = False

                if epoch >= num_epochs:
                    flag = False
                epoch = epoch + 1

        self.cwdn = load_model(f'torch_models//{self.model_name}_mixcwdn_{self.suffix}.pth', self.cwdn)

    def generate_results(self):
        model = self.cwdn
        train_preds = predict(model, self.train_dataset, self.device)
        evaluate(self.train_dataset.targets, train_preds.reshape(-1, 1), continuous=False, separator=0)
        test_preds = predict(model, self.test_dataset, self.device)
        evaluate(self.test_dataset.targets, test_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)
        val_preds = predict(model, self.val_dataset, self.device)
        evaluate(self.val_dataset.targets, val_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)

    def generate_test_stats(self):
        model = self.cwdn
        val_preds = predict(model, self.val_dataset, self.device)
        scores = plot_stats_box(self.val_dataset.targets, val_preds.reshape(-1, 1), groups=20)
        return scores


class AeSvm:
    def __init__(self, model_name, target, time_frame, suffix, batch_size, latent_dim=30, nca_dim=20, onca_c=3, kl_c=1,
                 max_epochs=12):
        self.model_name = model_name
        self.target = target
        self.timeframe = time_frame
        self.suffix = suffix
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.nca_dim = nca_dim
        self.onca_c = onca_c
        self.kl_c = kl_c
        self.max_epochs = max_epochs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.device)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.input_dim = None

        self.train_loader = None
        self.test_loader = None

        self.vae = None
        self.encoder = None
        self.classifier = None

        self.train_encoded = None
        self.test_encoded = None
        self.val_encoded = None

        self.generate_prerequisite()

    def generate_prerequisite(self):
        self.generate_input_dim_and_datasets()
        self.generate_data_loaders()
        self.generate_autoencoder()

    def generate_input_dim_and_datasets(self):
        data_loader = TrainTestValidationLoader(self.model_name, target=self.target, training_portion=0.75,
                                                validation_portion=0.4, original_time_frame=240)

        train_features, train_labels = data_loader.get_reframed_train_data()
        self.input_dim = np.shape(train_features)[2]
        self.train_dataset = ConvCandlesDataset(train_features, train_labels.reshape(-1, 1))

        test_features, test_labels = data_loader.get_reframed_test_data()
        self.test_dataset = ConvCandlesDataset(test_features, test_labels.reshape(-1, 1))

        val_features, val_labels = data_loader.get_reframed_val_data()
        self.val_dataset = ConvCandlesDataset(val_features, val_labels.reshape(-1, 1))

    def generate_data_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def generate_autoencoder(self):
        torch.manual_seed(0)

        self.vae = VariationalAutoencoder(self.input_dim, latent_dim=self.latent_dim, nca_dim=self.nca_dim,
                                          device=self.device, onca_c=self.onca_c, kl_c=self.kl_c)

        _ = self.vae.to(self.device, dtype=torch.double)

    def train_auto_encoder(self, lr=2e-3, weight_decay=5e-3):
        adam_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = np.inf
        patience = 3
        last_improvement = 0

        num_epochs = self.max_epochs
        epoch = 1
        flag = True
        while flag is True:
            start_time = time.time()

            total_loss, recon_loss, onca, kl = vae_train_epoch(self.vae, self.device, self.train_loader, adam_optimizer)
            total_loss2, recon_loss2, onca2, kl2 = vae_test_epoch(self.vae, self.device, self.test_loader)

            if epoch % 1 == 0:
                print('\n EPOCH {}/{} \t total_train {:.3f} \t reconstruction loss {:.3f} \t onca {:.5f} \t kl {:.3f}'
                      ''.format(epoch, num_epochs, total_loss, recon_loss, onca, kl))
                print('\t\t total_test {:.3f} \t reconstruction loss {:.3f} \t onca {:.5f} \t kl {:.3f} \t '
                      'execution time: {:.0f}s'.format(total_loss2, recon_loss2, onca2, kl2, time.time() - start_time))

            if total_loss2 < best_loss:
                save_vae_on_validation_improvement(self.vae, adam_optimizer, epoch,
                                                   f'{self.model_name}_vae_{self.suffix}')
                best_loss = total_loss2
                last_improvement = epoch

            if epoch - last_improvement >= patience:
                flag = False

            if epoch >= num_epochs:
                flag = False
            epoch = epoch + 1

    def generate_svm(self, kernel='rbf', c=1.0, gamma='scale'):
        self.classifier = SVC(kernel=kernel, C=c, gamma=gamma)
        self.encoder = VariationalEncoder(self.input_dim, latent_dim=self.latent_dim, nca_dim=self.nca_dim,
                                          device=self.device)
        self.encoder = load_encoder(f'torch_models//{self.model_name}_vae_{self.suffix}.pth', self.encoder)
        self.encoder.to(self.device, dtype=torch.double)

    def clear_the_encoded_data(self):
        self.train_encoded = None
        self.test_encoded = None
        self.val_encoded = None

    def train_svm(self):
        if self.train_encoded is None:
            train_preds = encode(self.encoder, self.train_dataset, self.device)
            x_train = train_preds[:, 0:self.nca_dim]
            self.train_encoded = x_train.reshape(np.shape(x_train)[0], np.shape(x_train)[2])

        y_train = self.train_dataset.targets.reshape(-1)
        self.classifier.fit(self.train_encoded, y_train)

    def generate_results(self):
        if self.train_encoded is None:
            train_preds = encode(self.encoder, self.train_dataset, self.device)
            x_train = train_preds[:, 0:self.nca_dim]
            self.train_encoded = x_train.reshape(np.shape(x_train)[0], np.shape(x_train)[2])

        train_preds = self.classifier.predict(self.train_encoded)
        evaluate(self.train_dataset.targets, train_preds.reshape(-1, 1), continuous=False, separator=0,
                 multiclass=False)

        if self.test_encoded is None:
            test_preds = encode(self.encoder, self.test_dataset, self.device)
            x_test = test_preds[:, 0:self.nca_dim]
            self.test_encoded = x_test.reshape(np.shape(x_test)[0], np.shape(x_test)[2])
        test_preds = self.classifier.predict(self.test_encoded)
        evaluate(self.test_dataset.targets, test_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)

        if self.val_encoded is None:
            val_preds = encode(self.encoder, self.val_dataset, self.device)
            x_val = val_preds[:, 0:self.nca_dim]
            self.val_encoded = x_val.reshape(np.shape(x_val)[0], np.shape(x_val)[2])
        val_preds = self.classifier.predict(self.val_encoded)
        evaluate(self.val_dataset.targets, val_preds.reshape(-1, 1), continuous=False, separator=0, multiclass=False)
