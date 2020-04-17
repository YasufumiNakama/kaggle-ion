# =====================================================================================
# Settings
# =====================================================================================
import os
ROOT = '../input/liverpool-ion-switching/'
CLEAN_ROOT = '../input/data-without-drift/'
OUTPUT_DIR = './output/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


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
    total_cate_size=10
    seed=42
    encoder='TRANSFORMER'
    target_size=11
    n_fold=2
    fold=[0, 1]
    model_type='seq2seq' #'seq2seq'


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
# FE
# =====================================================================================
def signal2cate(X_train, X_test=None, NUM_BINS=1000):
    signal_bins = np.linspace(X_train['signal'].min(), X_train['signal'].max(), NUM_BINS + 1)
    train_signal_dig = np.digitize(X_train['signal'], bins=signal_bins) - 1 
    train_signal_dig = np.minimum(train_signal_dig, len(signal_bins) - 2)
    X_train['signal_cate'] = train_signal_dig
    if X_test is not None:
        test_signal_dig = np.digitize(X_test['signal'], bins=signal_bins) - 1 
        test_signal_dig = np.minimum(test_signal_dig, len(signal_bins) - 2)
        X_test['signal_cate'] = test_signal_dig
        return  X_train, X_test
    return X_train

X_train = signal2cate(X_train, X_test=None, NUM_BINS=CFG.total_cate_size)


def rolling_features(df):

    window_sizes = [1000, 2000, 5000, 9000]

    for window in window_sizes:
        rolling = df['signal'].rolling(window=window)
        df["signal_rolling_std_" + str(window)] = rolling.std()
        
    return df

#X_train = rolling_features(X_train)


# =====================================================================================
# Dataset
# =====================================================================================
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df.copy()  
        self.target = df[TARGET].values
        self.cfg = cfg
        self.seq_len = self.cfg.seq_len
        self.cont_cols = self.cfg.cont_cols
        self.cate_cols = self.cfg.cate_cols
        self.cont_df = self.df[self.cont_cols]
        self.cate_df = self.df[self.cate_cols]
        
    def __getitem__(self, idx):
        if idx//500000 == (idx - self.seq_len//2)//500000:
            start_index = idx - self.seq_len//2
        else:
            start_index = idx//500000*500000
        if idx//500000 == (idx + self.seq_len//2)//500000:
            end_index = idx + self.seq_len//2
        else:
            end_index = (idx + self.seq_len//2)//500000*500000
        indices = self.df.iloc[start_index:end_index].index.tolist()
        seq_len = min(self.seq_len, len(indices))
        
        tmp_cont_x = torch.FloatTensor(self.cont_df.iloc[indices].values)
        cont_x = torch.FloatTensor(self.seq_len, len(self.cont_cols)).zero_()
        cont_x[-seq_len:] = tmp_cont_x[-seq_len:]
        
        tmp_cate_x = torch.LongTensor(self.cate_df.iloc[indices].values)
        cate_x = torch.LongTensor(self.seq_len, len(self.cate_cols)).zero_()
        cate_x[-seq_len:] = tmp_cate_x[-seq_len:]

        mask = torch.ByteTensor(self.seq_len).zero_()
        mask[-seq_len:] = 1
        
        label = self.target[idx]
        target = np.zeros(self.seq_len) - 1
        target[-seq_len:] = self.target[start_index:end_index]

        return cate_x, cont_x, mask, target, label

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
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
        self.cont_emb = nn.Sequential(                
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
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
        
    def forward(self, cate_x, cont_x, mask):
        batch_size = cate_x.size(0)

        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)
        cont_emb = self.cont_emb(cont_x)

        seq_emb = torch.cat([cate_emb, cont_emb], 2)
        
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
        
        if self.cfg.model_type == 'seq2seq':
            pred_y = self.reg_layer(sequence_output)
        else:
            pred_y = self.reg_layer(sequence_output[:, self.cfg.seq_len//2])
        return pred_y


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
    seq_len = model.cfg.seq_len
    model_type = model.cfg.model_type

    # switch to train mode
    model.train()
    train_preds, train_true = [], []

    start = end = time.time()
    global_step = 0
    
    for step, (cate_x, cont_x, mask, y, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        cate_x = cate_x.to(device)
        cont_x, mask, y = cont_x.to(device), mask.to(device), y.to(device)
        label = label.to(device)
        batch_size = cont_x.size(0)        
        
        # compute loss
        pred = model(cate_x, cont_x, mask)

        if model_type == 'seq2seq':
            pred_ = pred.view(-1, pred.shape[-1])
            y_ = y.view(-1)
            loss = L.FocalLoss(ignore_index=-1)(pred_, y_)
            train_true.append(label.detach().cpu().numpy())
            train_preds.append(pred[:, seq_len // 2, :].detach().cpu().numpy().argmax(1))
        else:
            loss = L.FocalLoss(ignore_index=-1)(pred, label)
            train_true.append(label.detach().cpu().numpy())
            train_preds.append(pred.detach().cpu().numpy().argmax(1))

        # record loss
        losses.update(loss.item(), batch_size)
        
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        
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
            # record accuracy
            score = f1_score(np.concatenate(train_preds), 
                             np.concatenate(train_true), labels=list(range(11)), average='macro')
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

    return losses.avg, scores.avg


def validate(valid_loader, model, device):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    sent_count = AverageMeter()
    seq_len = model.cfg.seq_len
    model_type = model.cfg.model_type
    
    # switch to evaluation mode
    model.eval()

    start = end = time.time()
    
    predictions = []
    groundtruth = []
    for step, (cate_x, cont_x, mask, y, label) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)        
        
        cate_x = cate_x.to(device)
        cont_x, mask, y = cont_x.to(device), mask.to(device), y.to(device)
        label = label.to(device)
        batch_size = cont_x.size(0)
        
        # compute loss
        with torch.no_grad():        
            pred = model(cate_x, cont_x, mask)        

            # record loss
            if model_type == 'seq2seq':
                pred_ = pred.view(-1, pred.shape[-1])
                y_ = y.view(-1)
                loss = L.FocalLoss(ignore_index=-1)(pred_, y_)
                predictions.append(pred[:, seq_len // 2, :].detach().cpu().numpy().argmax(1))
                groundtruth.append(label.detach().cpu().numpy())
            else:
                loss = L.FocalLoss(ignore_index=-1)(pred, label)
                predictions.append(pred.detach().cpu().numpy().argmax(1))
                groundtruth.append(label.detach().cpu().numpy())

            losses.update(loss.item(), batch_size)
        
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps    

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            # record accuracy
            score = f1_score(np.concatenate(predictions), 
                             np.concatenate(groundtruth), labels=list(range(11)), average='macro')
            scores.update(score, batch_size)

            print('TEST: {0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Score: {score.val:.4f}({score.avg:.4f}) '
                  'sent/s {sent_s:.0f} '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,                   
                   data_time=data_time, loss=losses,
                   score=scores,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   sent_s=sent_count.avg/batch_time.avg
                   ))

    predictions = np.concatenate(predictions)
    groundtruth = np.concatenate(groundtruth)
        
    # scoring
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
# Sampler
# =====================================================================================
from torch.utils.data.sampler import Sampler


class BalanceSampler(Sampler):
    def __init__(self, df, balance_col):
        self.length = len(df)
        self.balance_col = balance_col
        self.num_class = df[balance_col].nunique()
        class_map_df = pd.DataFrame({balance_col: df[balance_col].unique()}).reset_index()
        class_map_df.columns = ['label', balance_col]
        self.class_map = dict(class_map_df[[balance_col, 'label']].values)
        group = []
        target_gb = df.groupby([TARGET])
        for k, _ in self.class_map.items():
            g = target_gb.get_group(k).index
            group.append(list(g))
            assert(len(g)>0)
        self.group = group

    def __iter__(self):
        index = []
        n = 0
        is_loop = True
        while is_loop:
            c = np.arange(self.num_class)
            np.random.shuffle(c)
            for t in c:
                i = np.random.choice(self.group[t])
                index.append(i)
                n += 1
                if n == self.length:
                    is_loop = False
                    break
        return iter(index)

    def __len__(self):
        return self.length


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
    cate_cols = ['signal_cate']
    cont_cols = ['signal']
    print('cont_cols:', cont_cols)
    print('cate_cols:', cate_cols)
    CFG.cont_cols = cont_cols
    CFG.cate_cols = cate_cols
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # =====================================================================================
    # run function
    # =====================================================================================
    def run(fold, trn_idx, val_idx):

        train_db = TrainDataset(X_train.loc[trn_idx].reset_index(drop=True), CFG)
        valid_db = TrainDataset(X_train.loc[val_idx].reset_index(drop=True), CFG)

        train_loader = DataLoader(train_db, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_db, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

        num_train_optimization_steps = int(len(train_db) / CFG.batch_size / CFG.gradient_accumulation_steps) * (CFG.num_train_epochs)
        print('num_train_optimization_steps:', num_train_optimization_steps)

        # =====================================================================================
        # Prepare model
        # =====================================================================================
        model = TransfomerModel(CFG)

        # =====================================================================================
        # Prepare optimizer
        # =====================================================================================
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
        # =====================================================================================
        # Prepare scheduler
        # =====================================================================================
        """
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.warmup_steps, 
                                                    num_training_steps=num_train_optimization_steps)
        """
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, max_lr=1e-3, epochs=CFG.num_train_epochs, steps_per_epoch=len(train_loader))

        # =====================================================================================
        # Prepare log utils
        # =====================================================================================
        def get_lr():
            return scheduler.get_lr()[0]

        log_df = pd.DataFrame(columns=(['EPOCH']+['LR']+['TRAIN_LOSS', 'TRAIN_SCORE']+['VALID_LOSS', 'VALID_SCORE']) )

        curr_lr = get_lr()

        print(f'initial learning rate: {curr_lr}')

        # =====================================================================================
        # Training loop
        # =====================================================================================
        best_score = 0
        best_model = None
        best_epoch = 0

        model_list = []

        model.to(device)

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
                logger.info(log_df.tail(5))

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
        model = TransfomerModel(CFG)
        checkpoint = torch.load(OUTPUT_DIR + curr_model_name, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        valid_loss, valid_score, predictions, groundtruth = validate(valid_loader, model, device)
        logger.info(f'[Fold{fold} Best Saved model] valid_loss:{valid_loss} valid_score:{valid_score}')

        return predictions, groundtruth

    # =====================================================================================
    # k-fold
    # =====================================================================================
    folds = pd.read_csv('../input/ion-folds/folds.csv')
    predictions, groundtruth = [], []
    for fold in CFG.fold:
        trn_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index
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

