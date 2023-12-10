import os
import random

import numpy as np
import pandas as pd
from transformers import AdamW

from script.adjust_learning import *
from script.dataloader import *
from script.model import *
# DeepBL Model and DeepBC Model in ablation experiments
# from script.CNN_model import *
# from script.linear_model import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # Set Random Seed
    seed_val = 41
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoches = 15
    # Dataset Path
    dataset_folder = 'Dataset'
    for folder_name in os.listdir(dataset_folder):
        # Select file path
        folder_path = os.path.join(dataset_folder, folder_name)
        train_file = os.path.join(folder_path, 'train.csv')
        test_file = os.path.join(folder_path, 'test.csv')
        # Train_dataset
        train_data = pd.read_csv(train_file)
        train_sentences = train_data["sequence"]
        train_labels = train_data["label"]
        # Test_dataset
        test_data = pd.read_csv(test_file)
        test_sentences = test_data["sequence"]
        test_labels = test_data["label"]
        # Transformed into token and add padding
        train_inputs, train_labels, test_inputs, test_labels = input_token(train_sentences, train_labels,
                                                                           test_sentences, test_labels)
        # Calculate the length after adding padding
        input_channel = len(train_inputs[1])
        train_dataset = MyDataset(train_inputs, train_labels)
        test_dataset = MyDataset(test_inputs, test_labels)
        trainloader = Data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        testloader = Data.DataLoader(test_dataset, batch_size=32, shuffle=True)
        # Loading model
        bert_blend_cnn = Bert_Blend_CNN(input_channel)
        bert_blend_cnn.to(device)
        # Select optimizer and loss function
        optimizer = AdamW(bert_blend_cnn.parameters(), lr=1.5e-5, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        print('%s:' % folder_name)
        for epoch in range(0, epoches):
            print(f'Starting epoch {epoch + 1}')
            print('Starting training')
            # corrcet_number, total_number, real positive number, real and predict both are positive number
            correct, total, pos_num, tp = 0, 0, 0, 0
            for i, batch in enumerate(trainloader):
                optimizer.zero_grad()
                # batch[0] is token embedding; batch[1] is real label
                batch = tuple(p.to(device) for p in batch)
                # Data input to the model
                pred = bert_blend_cnn(batch[0])
                # Calculate loss function
                loss = loss_fn(pred, batch[1])
                # Back Propagation
                loss.backward()
                # Warm-up and Learning Rate Decay
                adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=epoches, lr_min=2e-6,
                                     lr_max=1.5e-5,
                                     warmup=True)
                # Model weight update
                optimizer.step()
                # predicte label
                _, predicted = torch.max(pred, 1)
                total += batch[1].size(0)
                # correct number
                correct += (predicted == batch[1]).sum().item()
                # positive number
                pos_num += (batch[1] == 1).sum().item()
                tp += ((batch[1] == 1) & (predicted == 1)).sum().item()
            neg_num = total - pos_num
            tn = correct - tp
            sn = tp / pos_num if pos_num != 0 else 1
            sp = tn / neg_num if neg_num != 0 else 1
            # Calculation accuracy
            acc = (tp + tn) / (pos_num + neg_num) if (pos_num + neg_num) != 0 else 1
            fn = pos_num - tp
            fp = neg_num - tn
            # Calculate Matthews correlation coefficient
            mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) \
                if (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) != 0 else 1
            print('Sn = %.4f  Sp = %.4f  Acc = %.4f  Mcc= %.4f  ' % (sn, sp, acc, mcc))
            print("train lr is ", optimizer.state_dict()["param_groups"][0]["lr"])
            print('Starting testing')
            # model valuing
            correct, total, pos_num, tp = 0, 0, 0, 0
            for i, batch in enumerate(testloader):
                batch = tuple(p.to(device) for p in batch)
                pred = bert_blend_cnn(batch[0])
                _, predicted = torch.max(pred, 1)
                total += batch[1].size(0)
                correct += (predicted == batch[1]).sum().item()
                pos_num += (batch[1] == 1).sum().item()
                tp += ((batch[1] == 1) & (predicted == 1)).sum().item()
            neg_num = total - pos_num
            tn = correct - tp
            sn = tp / pos_num if pos_num != 0 else 1
            sp = tn / neg_num if neg_num != 0 else 1
            acc = (tp + tn) / (pos_num + neg_num) if (pos_num + neg_num) != 0 else 1
            fn = pos_num - tp
            fp = neg_num - tn
            mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) \
                if (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) != 0 else 1
            print('Sn = %.4f  Sp = %.4f  Acc = %.4f  Mcc= %.4f ' % (sn, sp, acc, mcc))
            print('--------------------------------')
            # saving model
            if (epoch + 1) % 15 == 0:
                print(f'epoch = {epoch + 1}. Saving trained model.')
                save_path = os.path.join(folder_path, 'model.pth')
                torch.save(bert_blend_cnn.state_dict(), save_path)
