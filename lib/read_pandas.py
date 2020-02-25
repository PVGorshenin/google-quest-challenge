import numpy as np
from tqdm import tqdm

def read_preds(train_date, start_epoch=0, finish_epoch=5, is_train=True):
    stage = ''
    if not is_train: stage = 'val_'
    train_preds_lst = []
    for i in tqdm_notebook(range(start_epoch, finish_epoch+1)):
        i_preds = np.genfromtxt(f'../data/result/{train_date}/preds/{stage}preds_{i}.csv', delimiter=',')
        train_preds_lst.append(i_preds)
    return train_preds_lst


def _get_target_cols():
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    target_columns = submission.columns.values[1:].tolist()
    return target_columns