import numpy as np
import os
import torch
from common_ import to_numpy
from datetime import datetime
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, tqdm_notebook


DEVICE = torch.device('cuda')
OUTDIR = "../data/result/"


class BertPredictorKfold(object):

    def __init__(self, model, train_loader, criterion, optimizer, split_rand_state, val_loader=None, epochs_count=1):
        self.criterion = criterion
        self.epochs_count = epochs_count
        self.fulldir = ''
        self.i_epoch = 0
        self.model = model
        self.split_rand_state = split_rand_state
        self.optimizer = optimizer
        assert isinstance(train_loader, list)
        self.train_loader_lst = train_loader
        self.val_loader_lst = val_loader

    def _make_containers(self):
        self.np_train_preds = np.zeros((self.train_loader_lst[self.i_fold].num, 30, self.epochs_count))
        self.np_train_targets = np.zeros((self.train_loader_lst[self.i_fold].num, 30))
        self.np_val_preds = np.zeros((self.val_loader_lst[self.i_fold].num, 30, self.epochs_count))
        self.np_val_targets = np.zeros((self.val_loader_lst[self.i_fold].num, 30))

    def fit(self):
        for i_loader in tqdm_notebook(range(len(self.train_loader_lst))):
            self.i_fold = i_loader
            self.i_epoch = 0
            self._make_containers()
            self.spearman_score_lst = []
            self.val_spearman_score_lst = []

            self.train_loader = self.train_loader_lst[i_loader]
            self.val_loader = self.val_loader_lst[i_loader]
            assert self.train_loader.num / self.val_loader.num > 2
            for i_epoch in range(self.epochs_count):
                name_prefix = '[{} / {}] '.format(i_epoch + 1, self.epochs_count)
                train_loss = self.do_epoch(name_prefix + 'Train:')
                if not self.val_loader is None:
                    val_loss = self.predict_n_save_val()
                self.i_epoch += 1
            np.savetxt(os.path.join(self.fulldir, f'val_scores_fold_{i_loader}.csv'), self.val_spearman_score_lst,
                       delimiter=",")
            np.savetxt(os.path.join(self.fulldir, f'train_scores_fold_{i_loader}.csv'), self.spearman_score_lst,
                       delimiter=",")
            self.i_epoch += 1


    def _cals_spearman(self, logits, batch_labels):
        spearman_lst = []
        for i in range(30):
            spearman_lst.append(np.nan_to_num(spearmanr(logits[:, i], batch_labels[:, i], axis=0).correlation))
        return np.mean(spearman_lst)

    def _acumulate_preds(self, logits, batch_label, df_idx):
        if self.i_epoch:
            self.np_train_preds[df_idx, :, self.i_epoch] = to_numpy(logits)
        else:
            self.np_train_preds[df_idx, :, self.i_epoch] = to_numpy(logits)
            self.np_train_targets[df_idx] = to_numpy(batch_label)

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _make_fulldir_n_subs(self):
        now = datetime.now()
        datename = "-".join([str(now.date())[5:], str(now.hour)])
        self.fulldir = os.path.join(OUTDIR, datename)
        if not os.path.isdir(self.fulldir):
            os.makedirs(os.path.join(self.fulldir, 'preds'))
            os.makedirs(os.path.join(self.fulldir, 'models'))
        with open(os.path.join(self.fulldir, 'meta.txt'), 'w') as meta_file:
            meta_file.writelines(f'train loader len --> {len(self.train_loader)} \n')
            meta_file.write(f'split_rand_state --> {self.split_rand_state}\n')
            meta_file.write(f'model tokenizer --> {self.model.tokenizer}\n')

    def _save_train_results(self):
        with open(os.path.join(self.fulldir, 'meta.txt'), 'a') as meta_file:
            current_lr = self._get_lr()
            meta_file.writelines(f'\n i_epoch --> {self.i_epoch}  lr on end --> {current_lr}')


        torch.save(self.model.state_dict(), os.path.join(self.fulldir, 'models', f'bert_model_fold{self.i_fold}_epoch{self.i_epoch}'))
        np.savetxt(os.path.join(self.fulldir, 'preds', f'preds_fold{self.i_fold}_epoch{self.i_epoch}.csv'), self.np_train_preds[:, :, self.i_epoch],
                   delimiter=",")

    def do_epoch(self, name=None, is_train=True):
        epoch_loss = 0
        name = name or ''
        self.model.train(is_train)
        batches_count = len(self.train_loader)
        if self.fulldir == '':
            self._make_fulldir_n_subs()
        with torch.autograd.set_grad_enabled(is_train):
            with tqdm(total=batches_count) as progress_bar:
                for i_batch, (df_idx, batch_id, batch_seg, batch_label) in enumerate(self.train_loader):
                    logits = self.model(batch_id.cuda(), batch_seg.cuda())
                    loss = self.criterion(logits.cpu(), batch_label.cpu())
                    epoch_loss += loss.item()
                    self._acumulate_preds(logits, batch_label, df_idx)

                    if ((i_batch % 700) == 0) & (i_batch != 0):
                        is_zero = self.np_train_preds[:, :, self.i_epoch].sum(1) == 0
                        spearman_score = self._cals_spearman(self.np_train_preds[~is_zero, :, self.i_epoch],
                                                             self.np_train_targets[~is_zero])
                        self.spearman_score_lst.append(spearman_score)
                        print(spearman_score)
                        if self.val_loader:  self.predict_n_save_val(save=False)

                    if self.optimizer:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    progress_bar.update()
                    progress_bar.set_description('{:>5s} Loss = {:.5f}'.format(name, loss.item()))

                progress_bar.set_description('{:>5s} Loss = {:.5f}'.format(name, epoch_loss / batches_count))
                self._save_train_results()
        return epoch_loss / batches_count

    def predict_n_save_val(self, save=True):
        with torch.no_grad():
            for df_idx, batch_id, batch_seg, batch_label in self.val_loader:
                logits = self.model(batch_id.cuda(), batch_seg.cuda())
                self.np_val_preds[df_idx, :,  self.i_epoch] = (to_numpy(logits))
                self.np_val_targets[df_idx] = to_numpy(batch_label)
            is_zero = self.np_val_preds[:, :, self.i_epoch].sum(1) == 0
            val_spearman_score = self._cals_spearman(self.np_val_preds[~is_zero, :, self.i_epoch],
                                                     self.np_val_targets[~is_zero])
            self.val_spearman_score_lst.append(val_spearman_score)
            np.savetxt(os.path.join(self.fulldir, 'preds', f'val_preds_fold{self.i_fold}_epoch_{self.i_epoch}.csv'),
                       self.np_val_preds[:, :, self.i_epoch],
                       delimiter=",")
            if save:
                is_zero = self.np_train_preds[:, :, self.i_epoch].sum(1) == 0
                spearman_score = self._cals_spearman(self.np_train_preds[~is_zero, :, self.i_epoch],
                                                     self.np_train_targets[~is_zero])
                self.spearman_score_lst.append(spearman_score)

                if not self.i_epoch:
                    np.savetxt(os.path.join(self.fulldir, 'preds',  f'val_labels_fold{self.i_fold}.csv'), self.np_val_targets,
                               delimiter=",")



