# =====================================================================================
# Directory Settings
# =====================================================================================
import os
ROOT = '../input/liverpool-ion-switching/'
CLEAN_ROOT = '../input/data-without-drift/'
OUTPUT_DIR = './output/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class CFG:
    learning_rate=1.0e-3
    batch_size=16
    num_workers=6
    print_freq=1000
    test_freq=1
    start_epoch=0
    num_train_epochs=100
    warmup_steps=30
    max_grad_norm=1000
    gradient_accumulation_steps=1
    weight_decay=0.01
    dropout=0.3
    emb_size=100
    hidden_size=64
    nlayers=2
    nheads=10
    seq_len=4000
    seed=1225
    encoder='Wavenet' #'TRANSFORMER', 'Wavenet', 'RNN'
    optimizer='Adam' #'AdamW', 'Adam'
    target_size=11
    n_fold=5
    fold=[0]

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

USE_APEX = True

if USE_APEX:
    from apex import amp

# =====================================================================================
# Utils
# =====================================================================================
def get_logger(filename=OUTPUT_DIR+'log'):
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
    X_train = load_df(path=df_path_dict['train'])
    #X_test = load_df(path=df_path_dict['test'])
    #sample_submission = load_df(path=df_path_dict['sample_submission'])


# =====================================================================================
# Preprocess
# =====================================================================================
#X_train = X_train.reset_index()
#X_train['batch'] = X_train['index'] // 500000
X_train['batch'] = X_train.index // 500000
X_train['batch'] = X_train['batch'].astype(int)


# =====================================================================================
# FE
# =====================================================================================
def rolling_features(df):

    window_sizes = [1000, 2000, 5000, 9000]

    for window in window_sizes:
        rolling = df['signal'].rolling(window=window)
        df["signal_rolling_std_" + str(window)] = rolling.std()
        
    return df

#X_train = rolling_features(X_train)


# create batches of 4000 observations
def batching(df, batch_size):
    #print(df)
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# get lead and lags features
def lag_with_pct_change(df, windows):
    for window in windows:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, batch_size):
    # create batches
    df = batching(df, batch_size = batch_size)
    # create leads and lags (1, 2, 3 making them 6 features)
    df = lag_with_pct_change(df, [1, 2, 3])
    # create signal ** 2 (this is the new feature)
    df['signal_2'] = df['signal'] ** 2
    return df

X_train = run_feat_engineering(X_train, batch_size=CFG.seq_len)


# =====================================================================================
# Dataset
# =====================================================================================
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, df, sample_indices, seq_len, cont_cols):
        self.df = df.copy()
        self.target = df[TARGET].values
        self.sample_indices = sample_indices
        self.seq_len = seq_len
        self.cont_cols = cont_cols
        self.cont_df = self.df[self.cont_cols]
        #self.cont_df = np.log1p(self.df[self.cont_cols])

    def __getitem__(self, idx):
        indices = self.sample_indices[idx]
        seq_len = min(self.seq_len, len(indices))

        tmp_cont_x = torch.FloatTensor(self.cont_df.iloc[indices].values)
        cont_x = torch.FloatTensor(self.seq_len, len(self.cont_cols)).zero_()
        cont_x[-seq_len:] = tmp_cont_x[-seq_len:]

        mask = torch.ByteTensor(self.seq_len).zero_()
        mask[-seq_len:] = 1

        target = np.zeros(self.seq_len) - 1
        target[-seq_len:] = self.target[indices]

        return cont_x, mask, target

    def __len__(self):
        return len(self.sample_indices)


# =====================================================================================
# Transformer Model
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
            3, # not used
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
        
        #seq_length = self.cfg.seq_len
        #batch_size = cont_x.size(0)
        #position_ids = torch.arange(seq_length, dtype=torch.long, device=cont_x.device)
        #position_ids = position_ids.unsqueeze(0).expand((batch_size, seq_length))
        #position_emb = self.position_emb(position_ids)
        #seq_emb = (seq_emb + position_emb)
        #seq_emb = self.ln(seq_emb)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers
        
        encoded_layers = self.encoder(seq_emb, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        
        pred_y = self.reg_layer(sequence_output)
        return pred_y

# =====================================================================================
# RNN Model
# https://www.kaggle.com/brandenkmurray/seq2seq-rnn-with-gru
# =====================================================================================
class Seq2SeqRnn(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size, num_layers=1, bidirectional=False, dropout=.3,
                 hidden_layers = [100, 200]):
        
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.output_size=output_size
        
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                           bidirectional=bidirectional, batch_first=True, dropout=0.3)

        # Input Layer
        if hidden_layers and len(hidden_layers):
            first_layer  = nn.Linear(hidden_size*2 if bidirectional else hidden_size, hidden_layers[0])

            # Hidden Layers
            self.hidden_layers = nn.ModuleList(
                [first_layer]+[nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)]
            )
            for layer in self.hidden_layers: nn.init.kaiming_normal_(layer.weight.data)   

            self.intermediate_layer = nn.Linear(hidden_layers[-1], self.input_size)
            # output layers
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data) 
           
        else:
            self.hidden_layers = []
            self.intermediate_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_siz, self.input_size)
            self.output_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data) 

        self.activation_fn = torch.relu
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        #x = x.permute(0,2,1)

        outputs, _ = self.rnn(x)
        
        x = self.dropout(self.activation_fn(outputs))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
            x = self.dropout(x)
            
        x = self.output_layer(x)

        return x


# =====================================================================================
# Wavenet Model
# https://www.kaggle.com/cswwp347724/wavenet-pytorch
# =====================================================================================
class wave_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=int((kernel_size + (kernel_size-1)*(dilation-1))/2), dilation=dilation)
        self.conv3 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=int((kernel_size + (kernel_size-1)*(dilation-1))/2), dilation=dilation)
        self.conv4 = nn.Conv1d(out_ch, out_ch, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res_x = x
        tanh = self.tanh(self.conv2(x))
        sig = self.sigmoid(self.conv3(x))
        res = tanh.mul(sig)
        x = self.conv4(res)
        x = res_x + x
        return x


class Wavenet(nn.Module):
    def __init__(self, basic_block=wave_block):
        super().__init__()
        self.basic_block = basic_block
        self.layer1 = self._make_layers(8, 16, 3, 12)
        self.layer2 = self._make_layers(16, 32, 3, 8)
        self.layer3 = self._make_layers(32, 64, 3, 4)
        self.layer4 = self._make_layers(64, 128, 3, 1)
        self.lstm = nn.LSTM(input_size=8, hidden_size=64, num_layers=2, 
                            bidirectional=True, batch_first=True, dropout=0.3)
        self.gru = nn.GRU(64*2, hidden_size=64, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.3)
        def get_reg():
            return nn.Sequential(
            nn.Linear(128*2, 128*2),
            nn.LayerNorm(128*2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128*2, 128*2),
            nn.LayerNorm(128*2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128*2, 11),
        )
        self.fc = get_reg()

    def _make_layers(self, in_ch, out_ch, kernel_size, n):
        dilation_rates = [2 ** i for i in range(n)]
        layers = [nn.Conv1d(in_ch, out_ch, 1)]
        for dilation in dilation_rates:
            layers.append(self.basic_block(out_ch, out_ch, kernel_size, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        # RNN
        h_lstm, _ = self.lstm(x)
        h_gru, _ = self.gru(h_lstm)
        # CNN
        x = x.permute(0, 2, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = x.permute(0, 2, 1)
        #print(x.shape)
        # CNN & RNN
        #x = x + outputs
        x = torch.cat((x, h_gru.permute(0, 2, 1)), 1).permute(0, 2, 1)
        #print(x.shape)
        # fc
        x = self.fc(x)
        return x


# =====================================================================================
# Train functions
# =====================================================================================
import math
from sklearn.metrics import f1_score

from pytorch_toolbelt import losses as L


def train(train_loader, model, optimizer, epoch, scheduler, device):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    sent_count = AverageMeter()

    # switch to train mode
    model.train()
    train_preds, train_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)

    start = end = time.time()
    global_step = 0

    for step, (cont_x, mask, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        cont_x, mask, y = cont_x.to(device), mask.to(device), y.to(device)
        batch_size = cont_x.size(0)

        # compute loss
        #pred = model(cont_x, mask)
        pred = model(cont_x)

        pred_ = pred.view(-1, pred.shape[-1])
        y_ = y.view(-1)
        loss = L.FocalLoss(ignore_index=-1)(pred_, y_)

        # record loss
        losses.update(loss.item(), batch_size)
        """
        train_true = torch.cat([train_true, y_.long()], 0)
        train_preds = torch.cat([train_preds, pred_], 0)
        """

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if USE_APEX:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}]'.format(epoch, step, len(train_loader)))

        """
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            # record accuracy
            score = f1_score(train_true.cpu().detach().numpy(), train_preds.cpu().detach().numpy().argmax(1), labels=list(range(11)), average='macro')
            scores.update(score, batch_size)

            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Score: {score.val:.4f}({score.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  'sent/s {sent_s:.0f} '
                  .format(
                   epoch, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   score=scores,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   lr=scheduler.get_lr()[0],
                   #lr=scheduler.optimizer.param_groups[0]['lr'],
                   sent_s=sent_count.avg/batch_time.avg
                   ))
        """
    #return losses.avg, scores.avg
    return losses.avg, 0


def validate(valid_loader, model, device):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    sent_count = AverageMeter()

    # switch to evaluation mode
    model.eval()
    val_preds, val_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)

    start = end = time.time()

    for step, (cont_x, mask, y) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        cont_x, mask, y = cont_x.to(device), mask.to(device), y.to(device)
        batch_size = cont_x.size(0)

        # compute loss
        with torch.no_grad():
            #pred = model(cont_x, mask)
            pred = model(cont_x)

            # record loss
            pred_ = pred.view(-1, pred.shape[-1])
            y_ = y.view(-1)
            loss = L.FocalLoss(ignore_index=-1)(pred_, y_)

            losses.update(loss.item(), batch_size)

        # record accuracy
        val_true = torch.cat([val_true, y_.long()], 0)
        val_preds = torch.cat([val_preds, pred_], 0)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('TEST: {0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  #'Score: {score.val:.4f}({score.avg:.4f}) '
                  'sent/s {sent_s:.0f} '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   #score=scores,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   sent_s=sent_count.avg/batch_time.avg
                   ))

    # scoring
    predictions = val_preds.cpu().detach().numpy().argmax(1)
    groundtruth = val_true.cpu().detach().numpy()
    score = f1_score(predictions, groundtruth, labels=list(range(11)), average='macro')

    return losses.avg, score, predictions, groundtruth


def save_checkpoint(state, model_path, model_filename, is_best=False):
    print('saving cust_model ...')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(state, os.path.join(model_path, model_filename))
    if is_best:
        torch.save(state, os.path.join(model_path, 'best_' + model_filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def adjust_learning_rate(optimizer, epoch):
    #lr  = CFG.learning_rate
    lr = (CFG.lr_decay)**(epoch//10) * CFG.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# =====================================================================================
# Get Sample function
# =====================================================================================
def get_sample_indices(df):
    sample_indices = []
    group_indices = []
    df_groups = df.groupby('group').groups
    for group_idx, indices in enumerate(df_groups.values()):
        sample_indices.append(indices.values)
        group_indices.append(group_idx)
    return np.array(sample_indices), group_indices


# =====================================================================================
# Train loop
# =====================================================================================
import copy
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


def main():

    # =====================================================================================
    # Settings
    # =====================================================================================
    cont_cols = [c for c in X_train.columns if c.find('signal')>=0]
    logger.info(f'cont_cols: {cont_cols}')
    CFG.cont_cols = cont_cols
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')


    # =====================================================================================
    # run function
    # =====================================================================================
    def run(fold, trn_idx, val_idx):

        train_samples = sample_indices[trn_idx]
        valid_samples = sample_indices[val_idx]

        train_db = TrainDataset(X_train, train_samples, CFG.seq_len, cont_cols)
        valid_db = TrainDataset(X_train, valid_samples, CFG.seq_len, cont_cols)

        train_loader = DataLoader(train_db, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_db, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

        num_train_optimization_steps = int(len(train_db) / CFG.batch_size / CFG.gradient_accumulation_steps) * (CFG.num_train_epochs)
        logger.info(f'num_train_optimization_steps: {num_train_optimization_steps}')

        # =====================================================================================
        # Prepare model
        # =====================================================================================
        if CFG.encoder=='TRANSFORMER':
            model = TransfomerModel(CFG)
        elif CFG.encoder=='Wavenet':
            model = Wavenet()
        else:
            model = Seq2SeqRnn(input_size=len(cont_cols),
                               seq_len=CFG.seq_len,
                               hidden_size=CFG.hidden_size,
                               output_size=CFG.target_size,
                               num_layers=CFG.nlayers,
                               hidden_layers=[CFG.hidden_size,CFG.hidden_size,CFG.hidden_size],
                               bidirectional=True, 
                               dropout=CFG.dropout)
        
        model.to(device)

        # =====================================================================================
        # Prepare optimizer
        # =====================================================================================
        if CFG.optimizer=='AdamW':
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters,
                        lr=CFG.learning_rate,
                        weight_decay=CFG.weight_decay,
                        )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
        

        # =====================================================================================
        # Apex
        # =====================================================================================
        if USE_APEX:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        

        # =====================================================================================
        # Prepare scheduler
        # =====================================================================================
        """
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        """
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, max_lr=CFG.learning_rate, epochs=CFG.num_train_epochs, steps_per_epoch=len(train_loader))

        # =====================================================================================
        # Prepare log utils
        # =====================================================================================
        def get_lr():
            return scheduler.get_lr()[0]

        log_df = pd.DataFrame(columns=(['EPOCH']+['LR']+['TRAIN_LOSS', 'TRAIN_SCORE']+['VALID_LOSS', 'VALID_SCORE']) )
        curr_lr = get_lr()
        logger.info(f'initial learning rate: {curr_lr}')

        # =====================================================================================
        # Training loop
        # =====================================================================================
        best_score = 0
        best_model = None
        best_epoch = 0

        model_list = []

        for epoch in range(CFG.start_epoch, CFG.num_train_epochs):

            # train for one epoch
            train_loss, train_score = train(train_loader, model, optimizer, epoch, scheduler, device)

            valid_loss, valid_score, _, _ = validate(valid_loader, model, device)

            curr_lr = get_lr()
            print(f'set the learning_rate: {curr_lr}')

            model_list.append(copy.deepcopy(model))
            if epoch % CFG.test_freq == 0 and epoch >= 0:
                log_row = {'EPOCH':epoch, 'LR':curr_lr,
                        'TRAIN_LOSS':train_loss, 'TRAIN_SCORE':train_score,
                        'VALID_LOSS':valid_loss, 'VALID_SCORE':valid_score,
                        }
                log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)
                logger.info('Current Scores')
                logger.info(log_df.tail(5))
                logger.info('Best Scores')
                logger.info(log_df.sort_values('VALID_SCORE').tail(5))

                batch_size = CFG.batch_size*CFG.gradient_accumulation_steps

                if (best_score < valid_score):
                    best_model = copy.deepcopy(model)
                    best_score = valid_score
                    best_epoch = epoch

        model_to_save = best_model.module if hasattr(best_model, 'module') else best_model  # Only save the cust_model it-self
        curr_model_name = (f'f-{fold}_b-{batch_size}_a-{CFG.encoder}_e-{CFG.emb_size}_h-{CFG.hidden_size}_'
                       f'd-{CFG.dropout}_l-{CFG.nlayers}_hd-{CFG.nheads}_s-{CFG.seed}_len-{CFG.seq_len}.pt')
        torch.save({
            'epoch': best_epoch + 1,
            'arch': 'transformer',
            'state_dict': model_to_save.state_dict(),
            'log': log_df,
            },
            OUTPUT_DIR+curr_model_name,
        )

        # check
        if CFG.encoder=='TRANSFORMER':
            model = TransfomerModel(CFG)
        elif CFG.encoder=='Wavenet':
            model = Wavenet()
        else:
            model = Seq2SeqRnn(input_size=1,
                               seq_len=CFG.seq_len,
                               hidden_size=CFG.hidden_size,
                               output_size=CFG.target_size,
                               num_layers=CFG.nlayers,
                               hidden_layers=[CFG.hidden_size,CFG.hidden_size,CFG.hidden_size],
                               bidirectional=True,
                               dropout=CFG.dropout)
        checkpoint = torch.load(OUTPUT_DIR+curr_model_name, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        valid_loss, valid_score, predictions, groundtruth = validate(valid_loader, model, device)
        logger.info(f'[Fold{fold} Best Saved model] valid_loss:{valid_loss} valid_score:{valid_score}')

        return predictions, groundtruth

    # =====================================================================================
    # k-fold
    # =====================================================================================
    #X_train['group'] = X_train['index'] // CFG.seq_len
    sample_indices, group_indices = get_sample_indices(X_train)
    skf = GroupKFold(n_splits=CFG.n_fold)
    splits = [x for x in skf.split(sample_indices, None, group_indices)]
    predictions, groundtruth = [], []
    for fold, (trn_idx, val_idx) in enumerate(splits):
        if fold in CFG.fold:
            with timer(f'##### Running Fold: {fold} #####'):
                _predictions, _groundtruth = run(fold, trn_idx, val_idx)
                predictions.append(_predictions)
                groundtruth.append(_groundtruth)
    predictions = np.concatenate(predictions)
    groundtruth = np.concatenate(groundtruth)
    score = f1_score(predictions, groundtruth, labels=list(range(11)), average='macro')
    logger.info(f'##### CV Score: {score} #####')


if __name__ == '__main__':
    main()

