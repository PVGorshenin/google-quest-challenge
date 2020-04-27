import torch
import pandas as pd
from me.lib.bert_dataset import BertDataset
from sklearn.model_selection import KFold
from sklearn.model_selection import  train_test_split
from transformers import BertTokenizer

MAX_LEN = 512
SEP_TOKEN_ID = 102
DATA_DIR = "../../data/"


def get_test_loader(target_cols, batch_size=4):
    df = pd.read_csv(f'{DATA_DIR}test.csv')
    ds_test = BertDataset(df, target_cols, train_mode=False, labeled=False)
    loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0,
                                         collate_fn=ds_test.collate_fn, drop_last=False)
    loader.num = len(df)
    return loader


def _get_no_split(dataset_class, df, target_cols, batch_size, tokenizer, tokenizer_class):
        ds_train = dataset_class(df, target_cols, tokenizer=tokenizer, tokenizer_class=tokenizer_class)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                                     collate_fn=ds_train.collate_fn, drop_last=True)
        return train_loader, None


def get_train_val_loaders(dataset_class, df, target_cols, batch_size=4, val_batch_size=8, split_ratio=.8,
                          random_state=100, tokenizer=BertTokenizer,  tokenizer_class='bert-base-uncased'):

    if split_ratio == 1:
        train_loader, val_loader = _get_no_split(dataset_class, df, target_cols, batch_size, tokenizer, tokenizer_class)
        return train_loader, val_loader

    train_df, val_df = train_test_split(df, test_size=(1-split_ratio), random_state=random_state)
    print(f'val_df.shape[0] --> {val_df.shape[0]}')
    ds_train = dataset_class(train_df, target_cols, tokenizer=tokenizer, tokenizer_class=tokenizer_class)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                               collate_fn=ds_train.collate_fn, drop_last=True)
    train_loader.num = len(train_df)

    val_ds = dataset_class(val_df, target_cols, train_mode=False, tokenizer=tokenizer, tokenizer_class=tokenizer_class)
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
        ds_train = QuestDataset(df.iloc[train_ind], target_cols, tokenizer_class=tokenizer)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                                   collate_fn=ds_train.collate_fn, drop_last=True)
        train_loader.num = df.iloc[train_ind].shape[0]

        ds_val = QuestDataset(df.iloc[val_ind], target_cols, tokenizer_class=tokenizer)
        val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=True, num_workers=0,
                                                   collate_fn=ds_val.collate_fn, drop_last=True)
        val_loader.num = df.iloc[val_ind].shape[0]

        train_loader_lst.append(train_loader)
        val_loader_lst.append(val_loader)
        print(train_loader.num, val_loader.num)
    return train_loader_lst, val_loader_lst




