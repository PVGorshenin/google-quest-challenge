from transformers import RobertaModel
from me.lib.bert_model import MyBertModelStacked


class MyRobertaModel(MyBertModelStacked):
    def __init__(self,  model_type=RobertaModel, n_classes=30, hidden_dim=768, dropout=.2, tokenizer='roberta-base'):
        super(MyBertModelStacked, self).__init__(model_type, n_classes, hidden_dim, dropout, tokenizer)

    def _get_attention(self, ids):
        return (ids != 1).type(ids.type())

    def forward(self, ids):
        attention_mask = self._get_attention(ids)
        layers, pool_out, hiddens = self.bert_model(input_ids=ids,
                                                   attention_mask=attention_mask)
        logits = self.fc1(pool_out)
        return logits
