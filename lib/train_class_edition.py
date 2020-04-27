import numpy as np
import os
import torch
from me.lib.common_ import to_numpy
from me.lib.custom_logger import CustomLogger
from scipy.stats import spearmanr
from tqdm import tqdm


DEVICE = torch.device('cuda')
OUTDIR = "../data/result/"


def _calc_spearman(logits, batch_labels):
    spearman_lst = []
    for i in range(30):
        spearman_lst.append(np.nan_to_num(spearmanr(logits[:, i], batch_labels[:, i], axis=0).correlation))
    return np.mean(spearman_lst)


class BertPredictor(object):

    def __init__(self, model, train_loader, criterion, optimizer, split_rand_state,  val_loader=None, epochs_count=1,
                 result_dir="./data/result/", num_labels=30, is_segments=True):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.i_epoch = 0
        self.epochs_count = epochs_count
        self.is_segments = is_segments
        if val_loader is not None:
            self.logger = CustomLogger(self, train_loader.num, val_loader.num, epochs_count, num_labels, _calc_spearman,
                                       result_dir, split_rand_state)
        else:
            self.logger = CustomLogger(self, train_loader.num, 0, epochs_count, num_labels, _calc_spearman,
                                       result_dir, split_rand_state)

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def fit(self):
        self.logger._make_resultdir_n_subdirs()
        self.logger._save_loaders_information()
        for i_epoch in range(self.epochs_count):
            name_prefix = '[{} / {}] '.format(i_epoch + 1, self.epochs_count)
            self.do_epoch(name_prefix + 'Train:')
            if self.val_loader is not None:
                self.predict_n_save_val()
            self.i_epoch += 1
        self.logger._save_metric_scores()
        np.savetxt(os.path.join(self.logger.result_dir, 'preds', f'val_labels.csv'),
                   self.logger.np_val_targets, delimiter=",")

    def do_epoch(self, name=None):
        epoch_loss = 0
        name = name or ''
        self.model.train(True)
        batches_count = len(self.train_loader)
        if self.logger.result_dir == '':
            self.logger._make_resultdir_n_subdirs()
        with torch.autograd.set_grad_enabled(True):
            with tqdm(total=batches_count) as progress_bar:
                for i_batch, (df_idx, batch_id, batch_seg, batch_label) in enumerate(self.train_loader):
                    if self.is_segments:
                        logits = self.model(batch_id.cuda(), batch_seg.cuda())
                    else:
                        logits = self.model(batch_id.cuda())
                    loss = self.criterion(logits.cpu(), batch_label.cpu())
                    epoch_loss += loss.item()
                    self.logger._accumulate_preds(logits, batch_label, df_idx)

                    if self.optimizer:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    progress_bar.update()
                    progress_bar.set_description('{:>5s} Loss = {:.5f}'.format(name, loss.item()))

                progress_bar.set_description('{:>5s} Loss = {:.5f}'.format(name, epoch_loss / batches_count))
                self.logger._save_train_results()
                self.logger._extend_metric_lst_by_epoch(is_train=True)
        return epoch_loss / batches_count

    def predict_n_save_val(self):
        self.model.eval()
        with torch.no_grad():
            for df_idx, batch_id, batch_seg, batch_label in self.val_loader:
                if self.is_segments:
                    logits = self.model(batch_id.cuda(), batch_seg.cuda())
                else:
                    logits = self.model(batch_id.cuda())
                self.logger.np_val_preds[df_idx, :, self.i_epoch] = (to_numpy(logits))
                self.logger.np_val_targets[df_idx] = to_numpy(batch_label)
            self.logger._extend_metric_lst_by_epoch(is_train=False)
            np.savetxt(os.path.join(self.logger.result_dir, 'preds', f'val_preds_{self.i_epoch}.csv'),
                       self.logger.np_val_preds[:, :, self.i_epoch], delimiter=",")

