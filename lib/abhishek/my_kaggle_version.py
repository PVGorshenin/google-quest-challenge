import sys

sys.path.insert(0, "../input/transformers/transformers-master/")

import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import tensorflow_hub as hub
import transformers
from keras.models import Model, load_model
from numpy.random import seed
from transformers import DistilBertModel, DistilBertTokenizer

seed(42)
tf.random.set_seed(42)
random.seed(42)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def fetch_vectors(string_list, batch_size=64):
    DEVICE = torch.device("cuda")
    tokenizer = DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
    model = DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
    model.to(DEVICE)

    fin_features = []
    for data in chunks(string_list, batch_size):
        tokenized = []
        for x in data:
            x = " ".join(x.strip().split()[:300])
            tok = tokenizer.encode(x, add_special_tokens=True)
            tokenized.append(tok[:512])

        max_len = 512
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded).to(DEVICE)
        attention_mask = torch.tensor(attention_mask).to(DEVICE)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        fin_features.append(features)

    fin_features = np.vstack(fin_features)
    return fin_features


USE_PATH = "../input/universalsentenceencoderlarge4/"
USEDIST_PATH = "../input/distiluse/"

test = pd.read_csv("../input/google-quest-challenge/test.csv").fillna("none")
sample = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
test_question_body_dense = fetch_vectors(test.question_body.values)
test_answer_dense = fetch_vectors(test.answer.values)
input_columns = ['question_title', 'question_body', 'answer']
embed = hub.load(USE_PATH)
batch_size = 4


embeddings_test = {}
for input_column in input_columns:
    print(input_column)
    test_text = test[input_column].str.replace('?', '.').str.replace('!', '.').tolist()
    curr_test_emb = []
    ind = 0
    while ind * batch_size < len(test_text):
        curr_test_emb.append(embed(test_text[ind * batch_size: (ind + 1) * batch_size])["outputs"].numpy())
        ind += 1
    embeddings_test[input_column + '_embedding'] = np.vstack(curr_test_emb)


X_test = np.hstack([item for k, item in embeddings_test.items()])
X_test = np.hstack((X_test, test_question_body_dense, test_answer_dense))

for ind in range(5):
    model_name = f'best_model_batch{ind}.h5'
    model = load_model(os.path.join(USEDIST_PATH, model_name))
    test_preds_lst.append(model.predict(X_test))

model = load_model(os.path.join(USEDIST_PATH, 'fullmodel'))
test_preds_lst.append(model.predict(X_test))
print(len(test_preds_lst))