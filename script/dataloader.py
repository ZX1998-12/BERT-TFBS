import torch
import torch.utils.data as Data
from transformers import AutoTokenizer


def input_token(train_sentences, train_labels, test_sentences, test_labels):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    train_inputs = [tokenizer(sentence, return_tensors='pt')["input_ids"].squeeze() for sentence in train_sentences]
    test_inputs = [tokenizer(sentence, return_tensors='pt')["input_ids"].squeeze() for sentence in test_sentences]
    inputs = train_inputs + test_inputs
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    train_inputs = inputs[:len(train_inputs)]
    test_inputs = inputs[len(train_inputs):]

    return train_inputs, train_labels, test_inputs, test_labels


class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        inputs = self.sentences[index]
        label = self.labels[index]
        return inputs, label
