import torch
import numpy as np
from transformers import BertTokenizer
from math import floor, ceil

MAX_LEN = 512
SEP_TOKEN_ID = 102
DATA_DIR = "../data/"

class BertDataset(torch.utils.data.Dataset):

    def __init__(self, df, target_cols, train_mode=True, labeled=True, tokenizer=BertTokenizer,
                 tokenizer_class='bert-base-uncased'):
        self.df = df
        self.target_cols = target_cols
        self.train_mode = train_mode
        self.labeled = labeled
        if "-base-cased" in tokenizer_class:
            self.tokenizer = tokenizer.from_pretrained(tokenizer_class, do_lower_case=False)
        else:
            self.tokenizer = tokenizer.from_pretrained(tokenizer_class)

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