import os

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from script.dataloader import *
from script.model import *
# BERT-TFBS-V1 model and BERT-TFBS-V2 model in ablation experiments
# from script.model_V1 import *
# from script.model_V2 import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Dataset Path
        dataset_folder = 'Dataset'
        total_acc, total_roc_auc, total_pr_auc = 0, 0, 0
        results_df = pd.DataFrame(columns=['Name', 'Accuracy', 'ROC AUC', 'PR AUC'])
        for folder_name in os.listdir(dataset_folder):
            # Select file path
            folder_path = os.path.join(dataset_folder, folder_name)
            train_file = os.path.join(folder_path, 'train.csv')
            test_file = os.path.join(folder_path, 'test.csv')
            model_file = os.path.join(folder_path, 'model.pth')
            # Train_dataset
            train_data = pd.read_csv(train_file)
            train_sentences = train_data["sequence"]
            train_labels = train_data["label"]
            # Test_dataset
            test_data = pd.read_csv(test_file)
            test_sentences = test_data["sequence"]
            test_labels = test_data["label"]
            # Transformed into token and add padding
            train_inputs, train_labels, test_inputs, test_labels = input_token(train_sentences,
                                                                               train_labels,
                                                                               test_sentences, test_labels)
            # Calculate the length after adding padding
            input_channel = len(train_inputs[1])
            test_dataset = MyDataset(test_inputs, test_labels)
            testloader = Data.DataLoader(test_dataset, batch_size=128, shuffle=True)
            # Loading model
            bert_blend_cnn = Bert_Blend_CNN(input_channel)
            bert_blend_cnn.load_state_dict(torch.load(model_file))
            bert_blend_cnn.to(device)
            # Start testing
            bert_blend_cnn.eval()
            correct, total, pos_num, tp = 0, 0, 0, 0
            for i, batch in enumerate(testloader):
                # batch[0] is token embedding; batch[1] is label
                batch = tuple(p.to(device) for p in batch)
                # Data input to the model
                pred = bert_blend_cnn(batch[0])
                # predicte label
                _, predicted = torch.max(pred, 1)
                # Statistical predicted and true values
                if i == 0:
                    Y_valid = batch[1]
                else:
                    v = batch[1]
                    Y_valid = torch.cat([v, Y_valid], dim=0)
                if i == 0:
                    Y_pred = pred[:, 1] / (pred[:, 0] + pred[:, 1])
                else:
                    p = pred[:, 1] / (pred[:, 0] + pred[:, 1])
                    Y_pred = torch.cat([p, Y_pred], dim=0)
                total += batch[1].size(0)
                correct += (predicted == batch[1]).sum().item()
                pos_num += (batch[1] == 1).sum().item()
                tp += ((batch[1] == 1) & (predicted == 1)).sum().item()
            Y_valid = Y_valid.cpu()
            Y_pred = Y_pred.cpu()
            neg_num = total - pos_num
            tn = correct - tp
            acc = (tp + tn) / (pos_num + neg_num)
            # calculate ROC AUC and PR AUC
            roc_auc = roc_auc_score(Y_valid.numpy(), Y_pred.numpy())
            pr_auc = average_precision_score(Y_valid.numpy(), Y_pred.numpy())
            results_df = results_df.append({'Name': folder_name, 'Accuracy': acc, 'ROC AUC': roc_auc, 'PR AUC': pr_auc},
                                           ignore_index=True)
            total_acc += acc
            total_roc_auc += roc_auc
            total_pr_auc += pr_auc
            print('%s:Acc = %.4f, ROC-AUC = %.4f, PR-AUC = %.4f' % (folder_name, acc, roc_auc, pr_auc))
        # Calculate the mean of evaluation metrics
        mean_total_acc = total_acc / 165
        mean_total_roc_auc = total_roc_auc / 165
        mean_total_pr_auc = total_pr_auc / 165
        print('mean_Acc = %.4f, mean_roc_auc = %.4f, mean_pr_auc = %.4f' % (
            mean_total_acc, mean_total_roc_auc, mean_total_pr_auc))
        results_df = results_df.append(
            {'Name': 'average', 'Accuracy': mean_total_acc, 'ROC AUC': mean_total_roc_auc, 'PR AUC': mean_total_pr_auc},
            ignore_index=True)
        # saving model
        results_df.to_csv('results.csv', index=False)
