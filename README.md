# BERT-TFBS: a novel BERT-based model for predicting transcription factor binding sites by transfer learning

## Table of Contents

1. [Introduction](#introduction)
2. [Python Environment](#python-environment)
3. [Project Structure](#Project-Structure)
   1. [Dataset](#Dataset)
   2. [Model](#Model)
   3. [script](#script)
---

## 1. Introduction

We present a novel deep learning model called BERT-TFBS which is designed for predicting transcription factor binding sites (TFBSs) solely based on DNA sequences.The model comprises a pre-trained BERT module (DNABERT-2), a convolutional neural network (CNN) module, a convolutional block attention module (CBAM), and an output module. Training and testing are conducted on 165 ENCODE ChIP-seq datasets, demonstrating the superior ability of the BERT-TFBS to predict TFBSs. Experimental results indicate that BERT-TFBS achieves an average accuracy of 0.851, a ROC-AUC of 0.919, and a PR-AUC of 0.920 for TFBS prediction. Here, we provide the code used for TFBS prediction with BERT-TFBS.


## 2. Python Environment

Python 3.9 and packages version:

- torch==1.12.0
- torchvision==0.13.0
- transformers==4.22.2
- numpy==1.22.4
- pandas==1.4.4
- scikit-learn==1.1.1

## 3. Project Structure

### 3.1 **Dataset**

   -For this study, we choose 165 ChIP-seq datasets generated by the Encyclopedia of DNA Elements (ENCODE) project as benchmark datasets, which encompass 29 different TFs from various cell lines. According to the work of Zeng et al, each of the datasets can be randomly divided into a training set (80\%) and the corresponding test set (20\%), where each positive sample is a 101 bp DNA sequence that has been experimentally confirmed to contain TFBSs, and each negative sample is the sequence that is obtained from a positive sequence through random permutations while preserving the nucleotide frequencies.

   - This folder contains only the Hepg2 FOSL2 ChIP-seq dataset from the 165 chip-seq datasets. The original files are named `train.data` and `test.data`. We processe the raw data into `train.csv` and `test.csv` using the `pre_data.py` script, making it convenient for subsequent training.

### 3.2 **Model**
   -  The overall architectures of BERT-TFBS which consists of a DNABERT-2 module, a CNN module, a CBAM, and an output module, can be visualized in the following diagram:
     
      ![Model Architecture](https://github.com/ZX1998-12/BERT-TFBS/raw/master/Model/model.jpg)

   - `model.pth` is the model which is trained on the Hepg2 FOSL2 ChIP-seq dataset and represents the BERT-TFBS model.
     
   - The pre-trained BERT model is available at Huggingface as `zhihan1996/DNABERT-2-117M`.
     
     To load the model from Huggingface, you can use the following code:
     
     ```python
     import torch
     from transformers import AutoTokenizer, AutoModel
     
     tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
     model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
     ```
   - To train the model, you can run the `train.py` script using the training dataset.
     
     ```shell
     # Example code to train the model
     python train.py
     ```
     
     You can also run the `test.py` to test the model

### 3.3 **script**
   - `dataloader.py` converts DNA sequences into token embedding.
   - `CBAM.py` is the CBAM which integrates channel attention and spatial attention mechanisms, enhancing the representation of local features by emphasizing important channels and spatial information.
   - `adjust_learning.py` implements the learning rate of the optimizer is adjusted by using warm-up and cosine annealing techniques.
   - `linear_model.py` is the BERT-TFBS-v1 variant model which is constructed by removing the CNN module, CBAM, and the convolutional layer in the output module from BERT-TFBS.
   - `CNN_model.py` is the BERT-TFBS-v2 variant model which is constructed by removing CBAM from BERT-TFBS.
   - `model.py` is the BERT-TFBS which consists of a DNABERT-2 module, a CNN module, a CBAM, and an output module.
