import numpy as np
import os
import pandas as pd
import torch
from datetime import datetime
from my_common import to_numpy, df_
from loader import get_loader, _get_target_cols
from model import QuestModel
from tqdm import tqdm
from torch import nn


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def create_model(model_file, is_cased):
    if is_cased:
        model = QuestModel(KAGGLE_MODEL_PATH_CASED)
    else:
        model = QuestModel(KAGGLE_MODEL_PATH_UNCASED)
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    return model


def predict_test(test_loader, model):
    print('predict test')
    test_preds_lst = []
    with torch.no_grad():
        for batch_id, batch_seg in test_loader:
            logits = model(batch_id.cuda(), batch_seg.cuda())
            test_preds_lst.append(to_numpy(nn.Sigmoid()(logits)))
    return test_preds_lst


def one_predict(i_model, model_path, test_loader, is_cased=False):
    model = create_model(model_path, is_cased)
    print(f'{i_model} model has been loaded from {model_path}')
    test_preds = predict_test(test_loader, model)
    res_df = df_(np.concatenate(test_preds))
    res_df.columns = target_cols
    print(f"{i_model} model preds --> {res_df['answer_type_reason_explanation'].mean()}")
    return test_preds


UNCASED_PATH = '../input/bertdropout/bert_model_epoch'
CASED_PATH = '../input/bertdropoutcased2/bert_model_epoch'
KFOLD_UNCASED_PATH = '../input/bertkfolduncased/bert_model_'
KFOLD_CASED_PATH = '../input/bertkfoldcased/bert_model_'
KAGGLE_UNCASED_PATH = '../input/bertkaggleuncased/bert_model_epoch'

KAGGLE_MODEL_PATH_UNCASED = '../input/pretrained-bert-models-for-pytorch/bert-base-uncased/'
KAGGLE_MODEL_PATH_CASED = '../input/pretrained-bert-models-for-pytorch/bert-base-cased/'

SAVE_PRED = 1

FULL_UNCASED = 1
FULL_CASED = 1
KFOLD_CASED = 1
KFOLD_UNCASED = 1
KAGGLE_UNCASED = 1

START_ITER = 2
FINISH_ITER = 6
ONLY_ITER = 5

target_cols = _get_target_cols()
test_loader = get_loader(target_cols, batch_size=4, stage='test',
                         tokenizer='../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt')
assert (len(test_loader) > 50)

test_preds_lst = []
if FULL_UNCASED:
    for i_model in tqdm(range(START_ITER, FINISH_ITER + 1)):
        model_path = UNCASED_PATH + str(i_model)
        test_preds = one_predict(i_model, model_path, test_loader)
        test_preds_lst.append(np.concatenate(test_preds))

if FULL_CASED:
    for i_model in tqdm(range(START_ITER, FINISH_ITER + 1)):
        model_path = CASED_PATH + str(i_model)
        test_loader = get_loader(target_cols, batch_size=4, stage='test',
                                 tokenizer='../input/pretrained-bert-models-for-pytorch/bert-base-cased-vocab.txt')
        test_preds = one_predict(i_model, model_path, test_loader, is_cased=True)
        test_preds_lst.append(np.concatenate(test_preds))

if KFOLD_UNCASED:
    for i_model in tqdm(range(START_ITER, FINISH_ITER + 1)):
        model_path = KFOLD_UNCASED_PATH + f'fold{i_model}_epoch0'
        test_loader = get_loader(target_cols, batch_size=4, stage='test')
        test_preds = one_predict(i_model, model_path, test_loader)
        test_preds_lst.append(np.concatenate(test_preds))

if KFOLD_CASED:
    for i_model in tqdm(range(START_ITER, FINISH_ITER + 1)):
        model_path = KFOLD_CASED_PATH + f'fold{i_model}_epoch0'
        test_loader = get_loader(target_cols, batch_size=4, stage='test',
                                 tokenizer='../input/pretrained-bert-models-for-pytorch/bert-base-cased-vocab.txt')
        test_preds = one_predict(i_model, model_path, test_loader, is_cased=True)
        test_preds_lst.append(np.concatenate(test_preds))

if KAGGLE_UNCASED:
    for i_model in tqdm(range(START_ITER, FINISH_ITER + 1)):
        model_path = KAGGLE_UNCASED_PATH + str(i_model)
        test_loader = get_loader(target_cols, batch_size=4, stage='test')
        test_preds = one_predict(i_model, model_path, test_loader)
        test_preds_lst.append(np.concatenate(test_preds))

# else:
#     test_preds = one_predict(ONLY_ITER)

if SAVE_PRED:
    res_df = df_(np.mean(test_preds_lst, axis=0))
    res_df.columns = target_cols
    test_df = pd.read_csv('../input/google-quest-challenge/test.csv')
    res_df['qa_id'] = test_df['qa_id'].values

    res_df['question_type_spelling'] = test_df['question_body'].map(lambda z: 'pronoun' in z).astype(int)

    now = datetime.now()
    res_df.to_csv(f'submission_{now.day}_{now.hour}.csv', index=False)
    res_df.to_csv(f'submission.csv', index=False)
