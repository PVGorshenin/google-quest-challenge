from me.lib.bert_dataset import QuestDataset
from me.lib.roberta_dataset import RobertaDataset
from transformers import BertModel, RobertaModel, DistilBertModel

def get_dataset_by_name(dataset_name):
    if dataset_name == 'RobertaDataset':
        return RobertaDataset
    if dataset_name == 'QuestDataset':
        return QuestDataset


def get_model_by_name(model_name):
    if model_name == 'BertModel':
        return BertModel
    if model_name == 'RobertaModel':
        return RobertaModel
    if model_name == 'DistilBertModel':
        return DistilBertModel


