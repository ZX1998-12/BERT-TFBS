import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Build linear module
class CNNNET(nn.Module):
    def __init__(self, input_channel):
        super(CNNNET, self).__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.linear1 = nn.Linear(input_channel * 768, input_channel)
        self.linear2 = nn.Linear(input_channel, 2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        return F.softmax(x, dim=1)

# Build BERT-TFBS-V1 module
class Bert_Blend_CNN(nn.Module):
    def __init__(self, input_channel):
        super(Bert_Blend_CNN, self).__init__()
        self.bert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.model = CNNNET(input_channel)

    def forward(self, X):
        outputs = self.bert(X)
        cls_embeddings = outputs[0]
        logits = self.model(cls_embeddings)
        return logits
