import torch
import os
from pytorch_pretrained_bert import BertTokenizer
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from math import floor, ceil
from sklearn.model_selection import GroupKFold, train_test_split

MAX_LEN = 512
SEP_TOKEN_ID = 102
DATA_DIR = "../data/"

class QuestDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, target_cols, train_mode=True, labeled=True, tokenizer='bert-base-uncased'):
        self.df = df
        self.target_cols = target_cols
        self.train_mode = train_mode
        self.labeled = labeled
        if tokenizer == "bert-base-cased":
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer, do_lower_case=False)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        token_ids, seg_ids = self.get_token_ids(row)
        if self.labeled:
            labels = self.get_label(row)
            return index, token_ids, seg_ids, labels
        else:
            return index, token_ids, seg_ids

    def __len__(self):
        return len(self.df)

    def trim_input(self, title, question, answer, max_sequence_length=MAX_LEN,
                   title_max_len=30, question_max_len=239, answer_max_len=239):
        title_tokenized = self.tokenizer.tokenize(title)
        question_tokenized = self.tokenizer.tokenize(question)
        answer_tokenized = self.tokenizer.tokenize(answer)

        title_len = len(title_tokenized)
        question_len = len(question_tokenized)
        answer_len = len(answer_tokenized)

        if (title_len + question_len + answer_len + 4) > max_sequence_length:

            if title_max_len > title_len:
                title_new_len = title_len
                answer_max_len = answer_max_len + floor((title_max_len - title_len) / 2)
                question_max_len = question_max_len + ceil((title_max_len - title_len) / 2)
            else:
                title_new_len = title_max_len

            if answer_max_len > answer_len:
                answer_new_len = answer_len
                question_new_len = question_max_len + (answer_max_len - answer_len)
            elif question_max_len > question_len:
                answer_new_len = answer_max_len + (question_max_len - question_len)
                question_new_len = question_len
            else:
                answer_new_len = answer_max_len
                question_new_len = question_max_len

            if title_new_len + answer_new_len + question_new_len + 4 != max_sequence_length:
                raise ValueError("New sequence length should be %d, but is %d"
                                 % (max_sequence_length, (title_new_len + answer_new_len + question_new_len + 4)))

            title_tokenized = title_tokenized[:title_new_len]
            question_tokenized = question_tokenized[:question_new_len]
            answer_tokenized = answer_tokenized[:answer_new_len]

        return title_tokenized, question_tokenized, answer_tokenized

    def get_token_ids(self, row):
        t_tokens, q_tokens, a_tokens = self.trim_input(row.question_title, row.question_body, row.answer)

        tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < MAX_LEN:
            token_ids += [0] * (MAX_LEN - len(token_ids))
        ids = torch.tensor(token_ids)
        seg_ids = self.get_seg_ids(ids)
        return ids, seg_ids

    def get_seg_ids(self, ids):
        seg_ids = torch.zeros_like(ids)
        seg_idx = 0
        first_sep = True
        for i, e in enumerate(ids):
            seg_ids[i] = seg_idx
            if e == SEP_TOKEN_ID:
                if first_sep:
                    first_sep = False
                else:
                    seg_idx = 1
        pad_idx = torch.nonzero(ids == 0)
        seg_ids[pad_idx] = 0

        return seg_ids

    def get_label(self, row):
        return torch.tensor(row[self.target_cols].values.astype(np.float32))

    def collate_fn(self, batch):
        df_idx = [x[0] for x in batch]
        token_ids = torch.stack([x[1] for x in batch])
        seg_ids = torch.stack([x[2] for x in batch])

        if self.labeled:
            labels = torch.stack([x[3] for x in batch])
            return df_idx, token_ids, seg_ids, labels
        else:
            return df_idx, token_ids, seg_ids


class QuestDatasetType(QuestDataset):

    def __getitem__(self, index):
        row = self.df.iloc[index]
        type_number = self.df['category'].iloc[index]
        token_ids, seg_ids = self.get_token_ids(row)
        if self.labeled:
            labels = self.get_label(row)
            return index, token_ids, seg_ids, type_number, labels
        else:
            return index, token_ids, seg_ids, type_number


def get_test_loader(target_cols, batch_size=4):
    df = pd.read_csv(f'{DATA_DIR}test.csv')
    ds_test = QuestDataset(df, target_cols, train_mode=False, labeled=False)
    loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0,
                                         collate_fn=ds_test.collate_fn, drop_last=False)
    loader.num = len(df)

    return loader


def _gkf():
    gkf = GroupKFold(n_splits=5).split(X=df.question_body, groups=df.question_body)
    for fold, (train_idx, valid_idx) in enumerate(gkf):
        if fold == ifold:
            df_train = df.iloc[train_idx]
            df_val = df.iloc[valid_idx]
            break


def get_train_val_loaders(df, target_cols, batch_size=4, val_batch_size=8, split_ratio=.8, random_state=100,
                          tokenizer='bert-base-uncased'):

    if split_ratio == 1:
        ds_train = QuestDataset(df, target_cols, tokenizer=tokenizer)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                                     collate_fn=ds_train.collate_fn, drop_last=True)
        return train_loader, None

    train_df, val_df = train_test_split(df, test_size=(1-split_ratio), random_state=random_state)
    print(val_df.shape[0])
    ds_train = QuestDataset(train_df, target_cols, tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                               collate_fn=ds_train.collate_fn, drop_last=True)
    train_loader.num = len(train_df)

    val_ds = QuestDataset(val_df, target_cols, train_mode=False, tokenizer=tokenizer)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0,
                                             collate_fn=val_ds.collate_fn, drop_last=False)
    val_loader.num = len(val_df)
    val_loader.df = val_df

    return train_loader, val_loader


def get_train_val_loaders_kfold(df, target_cols, batch_size=4, val_batch_size=8, random_state=100, n_splits=5,
                                tokenizer='bert-base-uncased'):
    train_loader_lst, val_loader_lst = [], []
    kfold = KFold(n_splits=n_splits, random_state=random_state)
    for train_ind, val_ind in kfold.split(df):
        ds_train = QuestDataset(df.iloc[train_ind], target_cols, tokenizer=tokenizer)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                                   collate_fn=ds_train.collate_fn, drop_last=True)
        train_loader.num = df.iloc[train_ind].shape[0]

        ds_val = QuestDataset(df.iloc[val_ind], target_cols, tokenizer=tokenizer)
        val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=True, num_workers=0,
                                                   collate_fn=ds_val.collate_fn, drop_last=True)
        val_loader.num = df.iloc[val_ind].shape[0]

        train_loader_lst.append(train_loader)
        val_loader_lst.append(val_loader)
        print(train_loader.num, val_loader.num)
    return train_loader_lst, val_loader_lst



# sub = pd.read_csv("../data/sample_submission.csv")
# target_columns = sub.columns[1:]
# train_loader, _ = get_train_val_loaders()
# next(iter(train_loader))

