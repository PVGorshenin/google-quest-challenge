import torch
from transformers import DistilBertModel
from me.lib.bert_model import MyBertModelStacked


class MyDistilBertModel(MyBertModelStacked):
    def __init__(self, model_type=DistilBertModel, n_classes=30, hidden_dim=768, dropout=.2,
                 model_subtype='bert-base-uncased'):
        super(MyDistilBertModel, self).__init__(model_type, n_classes, hidden_dim, dropout, model_subtype)

    def forward(self, ids):
        attention_mask = self._get_attention(ids)
        last_hidden,  hiddens = self.bert_model(input_ids=ids,
                                                   attention_mask=attention_mask)
        hidden_layers = torch.stack([layer[:, 0, :] for layer in hiddens], dim=2)
        stacked_out = (hidden_layers * torch.softmax(self.layer_weights, dim=0)).sum(-1)
        logits = self.fc1(stacked_out)
        return logits