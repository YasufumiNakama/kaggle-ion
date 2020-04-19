# =====================================================================================
# Directory Settings
# =====================================================================================
ROOT = '../input/liverpool-ion-switching/'
CLEAN_ROOT = '../input/data-without-drift/'
OUTPUT_DIR = './'
MODEL_DIR = './'


# =====================================================================================
# Library
# =====================================================================================
import sys
import os
import gc
import time
from contextlib import contextmanager
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import random
import json

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn import preprocessing

from tqdm import tqdm, tqdm_notebook

import torch

import warnings
warnings.filterwarnings("ignore")

# =====================================================================================
# Utils
# =====================================================================================
def get_logger(filename=OUTPUT_DIR+'inference_log'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = get_logger()


@contextmanager
def timer(name):
    t0 = time.time()
    logger.info(f'[{name}] start')
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_df(path, debug=False):
    # load df .csv or .pkl
    if path.split('.')[-1]=='csv':
        df = pd.read_csv(path)
        if debug:
            df = pd.read_csv(path, nrows=1000)
    elif path.split('.')[-1]=='pkl':
        df = pd.read_pickle(path)
    print(f"{path} shape / {df.shape} ")
    return df


# =====================================================================================
# General Settings
# =====================================================================================
df_path_dict = {'train': CLEAN_ROOT+'train_clean.csv',
                'test': CLEAN_ROOT+'test_clean.csv',
                'sample_submission': ROOT+'sample_submission.csv'}
ID = 'time'
TARGET = 'open_channels'
SEED = 42
seed_everything(seed=SEED)


# =====================================================================================
# Data Loading
# =====================================================================================
with timer('Data Loading'):
    #X_train = load_df(path=df_path_dict['train'])
    X_test = load_df(path=df_path_dict['test'])
    submission = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype={'time': str})
    #X_test = X_test.head(10000)
    #submission = submission.head(10000)

# =====================================================================================
# FE
# =====================================================================================
def rolling_features(df):
    window_sizes = [1000, 2000, 5000, 9000]

    for window in window_sizes:
        rolling = df['signal'].rolling(window=window)
        df["signal_rolling_std_" + str(window)] = rolling.std()

    return df

# X_train = rolling_features(X_train)
# X_test = rolling_features(X_test)


# =====================================================================================
# Dataset
# =====================================================================================
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, df, seq_len, cont_cols):
        self.df = df.copy()
        self.target = df[TARGET].values
        self.seq_len = seq_len
        self.cont_cols = cont_cols
        self.cont_df = self.df[self.cont_cols]

    def __getitem__(self, idx):
        if idx // 500000 == (idx - self.seq_len // 2) // 500000:
            start_index = idx - self.seq_len // 2
        else:
            start_index = idx // 500000 * 500000
        if idx // 500000 == (idx + self.seq_len // 2) // 500000:
            end_index = idx + self.seq_len // 2
        else:
            end_index = (idx + self.seq_len // 2) // 500000 * 500000
        indices = self.df.iloc[start_index:end_index].index.tolist()
        seq_len = min(self.seq_len, len(indices))

        tmp_cont_x = torch.FloatTensor(self.cont_df.iloc[indices].values)
        cont_x = torch.FloatTensor(self.seq_len, len(self.cont_cols)).zero_()
        cont_x[-seq_len:] = tmp_cont_x[-seq_len:]

        mask = torch.ByteTensor(self.seq_len).zero_()
        mask[-seq_len:] = 1

        label = self.target[idx]
        target = np.zeros(self.seq_len) - 1
        target[-seq_len:] = self.target[start_index:end_index]

        return cont_x, mask, target, label

    def __len__(self):
        return len(self.df)


class TestDataset(Dataset):
    def __init__(self, df, seq_len, cont_cols):
        self.df = df.copy()
        self.seq_len = seq_len
        self.cont_cols = cont_cols
        self.cont_df = self.df[self.cont_cols]

    def __getitem__(self, idx):
        if idx // 500000 == (idx - self.seq_len // 2) // 500000:
            start_index = idx - self.seq_len // 2
        else:
            start_index = idx // 500000 * 500000
        if idx // 500000 == (idx + self.seq_len // 2) // 500000:
            end_index = idx + self.seq_len // 2
        else:
            end_index = (idx + self.seq_len // 2) // 500000 * 500000
        indices = self.df.iloc[start_index:end_index].index.tolist()
        seq_len = min(self.seq_len, len(indices))

        tmp_cont_x = torch.FloatTensor(self.cont_df.iloc[indices].values)
        cont_x = torch.FloatTensor(self.seq_len, len(self.cont_cols)).zero_()
        cont_x[-seq_len:] = tmp_cont_x[-seq_len:]

        mask = torch.ByteTensor(self.seq_len).zero_()
        mask[-seq_len:] = 1

        return cont_x, mask

    def __len__(self):
        return len(self.df)


# =====================================================================================
# Model
# =====================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertConfig, BertEncoder, BertModel


class TransfomerModel(nn.Module):
    def __init__(self, cfg):
        super(TransfomerModel, self).__init__()
        self.cfg = cfg
        cont_col_size = len(cfg.cont_cols)
        self.cont_emb = nn.Sequential(
            nn.Linear(cont_col_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
        )
        self.position_emb = nn.Embedding(num_embeddings=self.cfg.seq_len, embedding_dim=cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size)
        self.config = BertConfig(
            3,  # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.encoder = BertEncoder(self.config)

        def get_reg():
            return nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.Dropout(cfg.dropout),
                nn.ReLU(),
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.Dropout(cfg.dropout),
                nn.ReLU(),
                nn.Linear(cfg.hidden_size, cfg.target_size),
            )

        self.reg_layer = get_reg()

    def forward(self, cont_x, mask):
        seq_emb = self.cont_emb(cont_x)

        # seq_length = self.cfg.seq_len
        # batch_size = cont_x.size(0)
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=cont_x.device)
        # position_ids = position_ids.unsqueeze(0).expand((batch_size, seq_length))
        # position_emb = self.position_emb(position_ids)
        # seq_emb = (seq_emb + position_emb)
        # seq_emb = self.ln(seq_emb)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers

        encoded_layers = self.encoder(seq_emb, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        if self.cfg.model_type == 'seq2seq':
            pred_y = self.reg_layer(sequence_output)
        else:
            pred_y = self.reg_layer(sequence_output[:, self.cfg.seq_len // 2])
        return pred_y


# =====================================================================================
# Inference function
# =====================================================================================
def inference(test_loader, model, device):
    seq_len = model.cfg.seq_len
    model_type = model.cfg.model_type

    # switch to evaluation mode
    model.eval()

    #predictions = []
    probs = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))

    for step, (cont_x, mask) in tk0:

        cont_x, mask = cont_x.to(device), mask.to(device)
        batch_size = cont_x.size(0)

        # compute loss
        with torch.no_grad():
            pred = model(cont_x, mask)

        # record accuracy
        if model_type == 'seq2seq':
            prob = pred[:, seq_len // 2, :].detach().cpu().numpy()
            probs.append(prob)
            #predictions.append(prob.argmax(1))
        else:
            prob = pred.detach().cpu().numpy()
            probs.append(prob)
            #predictions.append(prob.argmax(1))

    probs = np.concatenate(probs)
    #predictions = np.concatenate(predictions)

    return probs#, predictions


# =====================================================================================
# Inference
# =====================================================================================
import copy
from torch.utils.data import DataLoader


class CFG:
    learning_rate=1.0e-4
    batch_size=256
    num_workers=6
    print_freq=1000
    test_freq=1
    start_epoch=0
    num_train_epochs=10
    warmup_steps=30
    max_grad_norm=1000
    gradient_accumulation_steps=1
    weight_decay=0.01    
    dropout=0.1
    emb_size=100
    hidden_size=100
    nlayers=2
    nheads=10
    seq_len=100
    seed=42
    encoder='TRANSFORMER'
    target_size=11
    n_fold=2
    fold=[0, 1]
    model_type='seq2seq' #'seq2seq'

batch_size = CFG.batch_size*CFG.gradient_accumulation_steps


def predict(model_path):
    # =====================================================================================
    # Settings
    # =====================================================================================
    # folds = pd.read_csv('../input/ion-folds/folds.csv')
    cont_cols = [c for c in X_test.columns if c.find('signal') >= 0]
    print('cont_cols:', cont_cols)
    CFG.cont_cols = cont_cols
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # =====================================================================================
    # Prepare loader
    # =====================================================================================
    test_db = TestDataset(X_test, CFG.seq_len, cont_cols)
    test_loader = DataLoader(test_db, batch_size=CFG.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # =====================================================================================
    # Prepare model
    # =====================================================================================
    model = TransfomerModel(CFG)
    checkpoint = torch.load(model_path, map_location=device)
    print(checkpoint['log'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    # =====================================================================================
    # Inference
    # =====================================================================================
    predictions = inference(test_loader, model, device)

    return predictions


def main():
    probs = []
    for fold in CFG.fold:
        curr_model_name = (f'f-{fold}_b-{batch_size}_a-{CFG.encoder}_e-{CFG.emb_size}_h-{CFG.hidden_size}_'
                           f'd-{CFG.dropout}_l-{CFG.nlayers}_hd-{CFG.nheads}_s-{CFG.seed}_len-{CFG.seq_len}.pt')
        model_path = MODEL_DIR + curr_model_name
        _probs = predict(model_path)
        probs.append(_probs)
    probs = np.mean(probs, axis=0)
    predictions = probs.argmax(1)
    sub = submission.copy()
    sub[TARGET] = predictions
    sub[TARGET] = sub[TARGET].astype(int)
    logger.info(sub.head())
    sub.to_csv(OUTPUT_DIR+'submission.csv', index=False)


if __name__ == '__main__':
    main()
