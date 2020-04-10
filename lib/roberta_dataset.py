import torch
from math import floor, ceil
from transformers import RobertaTokenizer
from me.lib.bert_dataset import QuestDataset

MAX_LEN = 497
SEP_TOKEN_ID = 102
DATA_DIR = "../data/"

class RobertaDataset(QuestDataset):
    def __init__(self, df, target_cols, train_mode=True, labeled=True, tokenizer=RobertaTokenizer,
                 tokenizer_class='bert-base-uncased'):
        self.df = df
        self.target_cols = target_cols
        self.train_mode = train_mode
        self.labeled = labeled
        self.n_specials = 7
        self.tokenizer = tokenizer.from_pretrained(tokenizer_class)

    def trim_input(self, title, question, answer, max_sequence_length=MAX_LEN,
                   title_max_len=30, question_max_len=230, answer_max_len=230):
        title_tokenized = self.tokenizer.tokenize(title)
        question_tokenized = self.tokenizer.tokenize(question)
        answer_tokenized = self.tokenizer.tokenize(answer)

        title_len = len(title_tokenized)
        question_len = len(question_tokenized)
        answer_len = len(answer_tokenized)

        if (title_len + question_len + answer_len + self.n_specials) > max_sequence_length:

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

            if title_new_len + answer_new_len + question_new_len + self.n_specials != max_sequence_length:
                raise ValueError("New sequence length should be %d, but is %d"
                                 % (max_sequence_length, (title_new_len + answer_new_len + question_new_len + self.n_specials)))

            title_tokenized = title_tokenized[:title_new_len]
            question_tokenized = question_tokenized[:question_new_len]
            answer_tokenized = answer_tokenized[:answer_new_len]

        return title_tokenized, question_tokenized, answer_tokenized

    def get_token_ids(self, row):
        t_tokens, q_tokens, a_tokens = self.trim_input(row.question_title, row.question_body, row.answer)

        tokens = [self.tokenizer.cls_token] + t_tokens + [self.tokenizer.sep_token] * 2 + q_tokens + \
                 [self.tokenizer.sep_token] * 2 + a_tokens + [self.tokenizer.sep_token] * 2
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < MAX_LEN:
            token_ids += [self.tokenizer.pad_token_id] * (MAX_LEN - len(token_ids))
        ids = torch.tensor(token_ids)
        seg_ids = self.get_seg_ids(ids)
        return ids, seg_ids

    def get_seg_ids(self, ids):
        seg_ids = torch.zeros_like(ids)
        seg_idx = 0
        first_sep = True
        for i, e in enumerate(ids):
            seg_ids[i] = seg_idx
            if e == self.tokenizer.sep_token_id:
                if first_sep:
                    first_sep = False
                else:
                    seg_idx = 1
        pad_idx = torch.nonzero(ids == self.tokenizer.pad_token_id)
        seg_ids[pad_idx] = 0
        return seg_ids