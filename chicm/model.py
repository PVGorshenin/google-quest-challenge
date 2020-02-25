from pytorch_pretrained_bert import BertModel
import torch
import torch.nn as nn


DEVICE = torch.device('cuda')


class QuestModel(nn.Module):
    def __init__(self, n_classes=30, hidden_dim=768, dropout=.2, tokenizer='bert-base-uncased'):
        super(QuestModel, self).__init__()
        self.model_name = 'QuestModel'
        self.tokenizer = tokenizer
        self.bert_model = BertModel.from_pretrained(tokenizer).to(DEVICE)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, seg_ids):
        attention_mask = (ids > 0)
        layers, pool_out = self.bert_model(input_ids=ids,
                                           token_type_ids=seg_ids,
                                           attention_mask=attention_mask)
        logits = self.fc1(self.dropout(pool_out))
        return logits
