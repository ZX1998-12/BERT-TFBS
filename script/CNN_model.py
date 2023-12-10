import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class CNNNET(nn.Module):
    def __init__(self, input_channel):
        super(CNNNET, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, dilation=2, padding=2, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_3 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, dilation=4, padding=4, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(180))
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(180),
            nn.GELU(),
            nn.Dropout(0.4))
        self.linear1 = nn.Linear(180 * 768, 180)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(180, 2)

    def forward(self, x):
        x = self.conv1(x)
        text1 = self.conv2_1(x)
        text2 = self.conv2_2(x)
        text3 = self.conv2_3(x)
        x = torch.cat([text1, text2, text3], dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)
        return F.softmax(x, dim=1)


class Bert_Blend_CNN(nn.Module):
    def __init__(self,input_channel):
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