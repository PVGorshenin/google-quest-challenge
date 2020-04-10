from transformers import BertModel
import torch
import torch.nn as nn


DEVICE = torch.device('cuda')


class MyBertModel(nn.Module):
    def __init__(self, model_type=BertModel, n_classes=30, hidden_dim=768, dropout=.2,
                 model_subtype='bert-base-uncased'):
        super(MyBertModel, self).__init__()
        self.model_name = 'QuestModel'
        self.bert_model = model_type.from_pretrained(model_subtype, output_hidden_states=True).to(DEVICE)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, n_classes)
        self.sigmoid = nn.Sigmoid()

    def _get_attention(self, ids):
        return ids > 0

    def forward(self, ids, seg_ids):
        attention_mask = self._get_attention(ids)
        layers, pool_out, hiddens = self.bert_model(input_ids=ids,
                                                   token_type_ids=seg_ids,
                                                   attention_mask=attention_mask)
        logits = self.fc1(pool_out)
        return logits


class MyBertModelStacked(MyBertModel):
    def __init__(self, model_type=BertModel, n_classes=30, hidden_dim=768, dropout=.2,
                 model_subtype='bert-base-uncased'):
        super(MyBertModelStacked, self).__init__(model_type, n_classes, hidden_dim, dropout, model_subtype)
        weights_init = torch.zeros(self.bert_model.config.num_hidden_layers + 1).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

    def forward(self, ids, seg_ids):
        attention_mask = self._get_attention(ids)
        layers, pool_out, hiddens = self.bert_model(input_ids=ids,
                                                   token_type_ids=seg_ids,
                                                   attention_mask=attention_mask)
        hidden_layers = torch.stack([layer[:, 0, :] for layer in hiddens], dim=2)
        stacked_out = (hidden_layers * torch.softmax(self.layer_weights, dim=0)).sum(-1)
        logits = self.fc1(stacked_out)
        return logits