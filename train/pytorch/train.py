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
    learning_rate=3.0e-3
    batch_size=32
    num_workers=6
    print_freq=1000
    test_freq=1
    start_epoch=0
    num_train_epochs=150
    warmup_steps=30
    max_grad_norm=1000
    gradient_accumulation_steps=1
    weight_decay=0.01
    dropout=0.3
    emb_size=100
    hidden_size=164
    nlayers=2
    nheads=10
    seq_len=5000
    total_cate_size=40
    seed=111
    encoder='Wavenet'
    optimizer='Adam' #@param ['AdamW', 'Adam']
    target_size=11
    n_fold=4
    fold=[0, 1, 2, 3]


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
from sklearn.cluster import KMeans

from tqdm import tqdm, tqdm_notebook

import torch

import warnings
warnings.filterwarnings("ignore")

USE_APEX = False

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


def make_stratified_group_k_folds(_df, _id, target, group, k, seed=42, save_path=OUTPUT_DIR+'folds.csv'):

    def stratified_group_k_fold(X, y, groups, k, seed=42):

        """
        original author : jakubwasikowski
        reference : https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
        """

        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(seed).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(k):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(k):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices

    df = _df.copy()
    le = preprocessing.LabelEncoder()
    groups = le.fit_transform(df[group].values)
    for n, (train_index, val_index) in enumerate(stratified_group_k_fold(df, df[target], groups, k=k, seed=seed)):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    df[[_id, target, group, 'fold']].to_csv(save_path, index=None)

    return df[[_id, target, group, 'fold']]


# =====================================================================================
# General Settings
# =====================================================================================
df_path_dict = {'train': CLEAN_ROOT+'train_clean.csv',
                'test': CLEAN_ROOT+'test_clean.csv',
                'sample_submission': ROOT+'sample_submission.csv'}
ID = 'time'
TARGET = 'open_channels'
seed_everything(seed=CFG.seed)


# =====================================================================================
# Data Loading
# =====================================================================================
with timer('Data Loading'):
    X_train = load_df(path=df_path_dict['train'])
    #X_test = load_df(path=df_path_dict['test'])
    #sample_submission = load_df(path=df_path_dict['sample_submission'])
    oof_lgb = pd.read_csv('../input/ion-oof/lgb_oof.csv').rename(columns={TARGET:'signal_lgb_oof'})
    oof_cat = pd.read_csv('../input/ion-oof/cat_oof.csv').rename(columns={TARGET:'signal_cat_oof'})

    X_train = pd.concat([X_train, oof_lgb['signal_lgb_oof']], axis=1)
    X_train = pd.concat([X_train, oof_cat['signal_cat_oof']], axis=1)

    del oof_lgb, oof_cat; gc.collect()

# =====================================================================================
# Preprocess
# =====================================================================================
X_train['batch'] = X_train.index // 500000
X_train['batch'] = X_train['batch'].astype(int)


from sklearn.cluster import KMeans

def categorize_batch(X_train, X_test=None):
    aggs = ['median']
    group_df = X_train.groupby('batch')['signal'].agg(aggs)
    group_df.columns = pd.Index(['signal_{}'.format(c) for c in group_df.columns.tolist()])
    cols = ['signal_median']
    print(f'cols: {cols}')
    kmeans = KMeans(n_clusters=5, random_state=42).fit(group_df[cols].values)
    group_df['batch_category'] = kmeans.labels_
    X_train = X_train.merge(group_df.reset_index()[['batch', 'batch_category']], on='batch', how='left')
    if X_test is not None:
        group_df = X_test.groupby('batch')['signal'].agg(aggs)
        group_df.columns = pd.Index(['signal_{}'.format(c) for c in group_df.columns.tolist()])
        group_df['batch_category'] = kmeans.predict(group_df[cols].values)
        X_test = X_test.merge(group_df.reset_index()[['batch', 'batch_category']], on='batch', how='left')
        return X_train, X_test
    return X_train

#X_train = categorize_batch(X_train, X_test=None)


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


def add_num_features(X_train, X_test=None):
    max_signal = X_train['signal'].max()
    min_signal = X_train['signal'].min()
    X_train['signal_diff_max'] = max_signal - X_train['signal']
    X_train['signal_diff_min'] = min_signal - X_train['signal']
    if X_test is not None:
        X_test['signal_diff_max'] = max_signal - X_test['signal']
        X_test['signal_diff_min'] = min_signal - X_test['signal']
        return  X_train, X_test
    return X_train

X_train = add_num_features(X_train, X_test=None)

def calc_gradients(df):

    df['signal_gradient'] = np.gradient(df['signal'].values)

    return df

def preprocess_df(df):

    output = pd.DataFrame()

    for i in range(int(len(df)/500000)):
        tmp = df.loc[i * 500000: 500000*(i + 1) - 1].reset_index(drop=True)
        tmp = calc_gradients(tmp)
        output = pd.concat([output, tmp])

    return output.reset_index(drop=True)

X_train = preprocess_df(X_train)


# =====================================================================================
# Dataset
# =====================================================================================
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, df, sample_indices, cfg):
        self.df = df.copy()
        self.target = df[TARGET].values
        self.sample_indices = sample_indices
        self.cfg = cfg
        self.seq_len = self.cfg.seq_len
        self.cont_cols = self.cfg.cont_cols
        self.cate_cols = self.cfg.cate_cols
        self.cont_df = self.df[self.cont_cols]
        self.cate_df = self.df[self.cate_cols]

    def __getitem__(self, idx):
        indices = self.sample_indices[idx]
        seq_len = min(self.seq_len, len(indices))

        tmp_cont_x = torch.FloatTensor(self.cont_df.iloc[indices].values)
        cont_x = torch.FloatTensor(self.seq_len, len(self.cont_cols)).zero_()
        cont_x[-seq_len:] = tmp_cont_x[-seq_len:]

        tmp_cate_x = torch.LongTensor(self.cate_df.iloc[indices].values)
        cate_x = torch.LongTensor(self.seq_len, len(self.cate_cols)).zero_()
        cate_x[-seq_len:] = tmp_cate_x[-seq_len:]

        target = np.zeros(self.seq_len) - 1
        target[-seq_len:] = self.target[indices]

        return cate_x, cont_x, target

    def __len__(self):
        return len(self.sample_indices)


# =====================================================================================
# Wavenet Model
# =====================================================================================
import torch.nn as nn
import torch.nn.functional as F


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
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        res_x = x
        tanh = self.tanh(self.conv2(x))
        sig = self.sigmoid(self.conv3(x))
        x = tanh.mul(sig)
        x = self.conv4(x)
        x = res_x + x
        return x


class Wavenet(nn.Module):
    def __init__(self, cfg, basic_block=wave_block):
        super().__init__()
        self.cfg = cfg
        self.basic_block = basic_block
        cont_col_size = len(cfg.cont_cols)
        cate_col_size = len(cfg.cate_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        #self.cate_proj = nn.Sequential(
        #    nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
        #    nn.LayerNorm(cfg.hidden_size//2),
        #)
        self.layer1 = self._make_layers(cont_col_size, cfg.hidden_size//16, 3, 12)
        self.layer2 = self._make_layers(cfg.hidden_size//16, cfg.hidden_size//8, 3, 8)
        self.layer3 = self._make_layers(cfg.hidden_size//8, cfg.hidden_size//4, 3, 4)
        self.layer4 = self._make_layers(cfg.hidden_size//4, cfg.hidden_size//2, 3, 1)
        self.gru = nn.GRU(input_size=cfg.emb_size*cate_col_size+cont_col_size, hidden_size=cfg.hidden_size//4, num_layers=cfg.nlayers,
                          bidirectional=True, batch_first=True, dropout=cfg.dropout)
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
        self.fc = get_reg()

    def _make_layers(self, in_ch, out_ch, kernel_size, n):
        dilation_rates = [2 ** i for i in range(n)]
        layers = [nn.Conv1d(in_ch, out_ch, 1)]
        for dilation in dilation_rates:
            layers.append(self.basic_block(out_ch, out_ch, kernel_size, dilation))
        return nn.Sequential(*layers)

    def forward(self, cate_x, cont_x):
        batch_size = cate_x.size(0)
        # RNN
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        seq_emb = torch.cat((cate_emb, cont_x), 2)
        h_gru, _ = self.gru(seq_emb)
        # CNN
        cont_x = cont_x.permute(0, 2, 1)
        cont_x = self.layer1(cont_x)
        cont_x = self.layer2(cont_x)
        cont_x = self.layer3(cont_x)
        cont_x = self.layer4(cont_x)
        # CNN & RNN
        x = torch.cat((cont_x, h_gru.permute(0, 2, 1)), 1).permute(0, 2, 1)
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

    for step, (cate_x, cont_x, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        cate_x, cont_x, y = cate_x.to(device), cont_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        # compute loss
        pred = model(cate_x, cont_x)

        pred_ = pred.view(-1, pred.shape[-1])
        y_ = y.view(-1)
        loss = L.FocalLoss(ignore_index=-1)(pred_, y_)
        #loss = nn.CrossEntropyLoss(ignore_index=-1)(pred_, y_.long())

        # record loss
        losses.update(loss.item(), batch_size)
        train_true = torch.cat([train_true, y_.long()], 0)
        train_preds = torch.cat([train_preds, pred_], 0)

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
                   sent_s=sent_count.avg/batch_time.avg
                   ))
    
    #optimizer.update_swa()
    #optimizer.swap_swa_sgd()

    predictions = train_preds.cpu().detach().numpy().argmax(1)
    groundtruth = train_true.cpu().detach().numpy()
    score = f1_score(predictions, groundtruth, labels=list(range(11)), average='macro')

    return losses.avg, score


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

    for step, (cate_x, cont_x, y) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        cate_x, cont_x, y = cate_x.to(device), cont_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        # compute loss
        with torch.no_grad():
            pred = model(cate_x, cont_x)

            # record loss
            pred_ = pred.view(-1, pred.shape[-1])
            y_ = y.view(-1)
            loss = L.FocalLoss(ignore_index=-1)(pred_, y_)
            #loss = nn.CrossEntropyLoss(ignore_index=-1)(pred_, y_.long())

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


# =====================================================================================
# Get Sample function
# =====================================================================================
def tta_group(X_train):
    X_train = X_train.reset_index()
    tmp = pd.DataFrame()
    for i in tqdm(range(int(len(X_train)/500000))):
        _tmp = X_train[i*500000+CFG.seq_len//2:(i+1)*500000-CFG.seq_len//2].reset_index(drop=True)
        tmp = pd.concat([tmp, _tmp])
    tmp = tmp.reset_index(drop=True)
    tmp['tta_group'] = tmp.index // CFG.seq_len
    tmp['tta_group'] = tmp['tta_group'] + X_train['group'].nunique()
    X_train = X_train.merge(tmp[['index', 'tta_group']], on='index', how='left').fillna(-1).drop(columns='index')
    return X_train


def get_sample_indices(df):
    sample_indices = []
    group_indices = []
    df_groups = df.groupby('group').groups
    tta_df_groups = df[df['tta_group']>=0].groupby('tta_group').groups
    for group_idx, indices in enumerate(df_groups.values()):
        sample_indices.append(indices.values)
        group_indices.append(group_idx)
    for group_idx, indices in enumerate(tta_df_groups.values()):
        sample_indices.append(indices.values)
        group_indices.append(group_idx+len(df_groups))
    return np.array(sample_indices), group_indices


# =====================================================================================
# Train loop
# =====================================================================================
import copy
from torch.utils.data import DataLoader
#import torchcontrib
#from torchcontrib.optim import SWA


def main(X_train, folds):

    # =====================================================================================
    # Settings
    # =====================================================================================
    #cate_cols = [c for c in X_train.columns if c.find('cate')>=0]
    cate_cols = ['signal_cate']
    cont_cols = [c for c in X_train.columns if c.find('signal')>=0]
    cont_cols = [c for c in cont_cols if c not in cate_cols]
    logger.info(f'cont_cols: {cont_cols}')
    logger.info(f'cate_cols: {cate_cols}')
    CFG.cont_cols = cont_cols
    CFG.cate_cols = cate_cols
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')


    # =====================================================================================
    # run function
    # =====================================================================================
    def run(fold, trn_idx, val_idx):

        train_samples = sample_indices[trn_idx]
        valid_samples = sample_indices[val_idx]

        train_db = TrainDataset(X_train, train_samples, CFG)
        valid_db = TrainDataset(X_train, valid_samples, CFG)

        train_loader = DataLoader(train_db, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_db, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

        num_train_optimization_steps = int(len(train_db) / CFG.batch_size / CFG.gradient_accumulation_steps) * (CFG.num_train_epochs)
        logger.info(f'num_train_optimization_steps: {num_train_optimization_steps}')

        # =====================================================================================
        # Prepare model
        # =====================================================================================
        model = Wavenet(CFG)
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
            #optimizer = torchcontrib.optim.SWA(optimizer, swa_start=10, swa_freq=2, swa_lr=1e-3)        
        
        # =====================================================================================
        # Apex
        # =====================================================================================
        if USE_APEX:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

        # =====================================================================================
        # Prepare scheduler
        # =====================================================================================
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, max_lr=CFG.learning_rate, epochs=CFG.num_train_epochs, steps_per_epoch=len(train_loader))
        #schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.2)
        
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
        model = Wavenet(CFG)
        checkpoint = torch.load(OUTPUT_DIR+curr_model_name, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        valid_loss, valid_score, predictions, groundtruth = validate(valid_loader, model, device)
        logger.info(f'[Fold{fold} Best Saved model] valid_loss:{valid_loss} valid_score:{valid_score}')

        return predictions, groundtruth

    # =====================================================================================
    # k-fold
    # =====================================================================================
    folds = tta_group(folds)
    folds['tta_group'] = folds['tta_group'].astype(int)
    sample_indices, group_indices = get_sample_indices(folds)
    print(len(group_indices))
    print(len(set(group_indices)))
    group_map = dict(folds[['group', 'split_group']].values.tolist())
    tta_group_map = dict(folds[['tta_group', 'split_group']].values.tolist())
    group_map.update(tta_group_map)
    group_indices = [group_map[i] for i in group_indices]
    print(len(group_indices))
    print(len(set(group_indices)))
    folds_map = dict(folds[['split_group', 'fold']].values.tolist())
    group_indices = [folds_map[i] for i in group_indices]
    skf = GroupKFold(n_splits=CFG.n_fold)
    splits = [x for x in skf.split(sample_indices, None, group_indices)]
    predictions, groundtruth = [], []
    for fold, (trn_idx, val_idx) in enumerate(splits):
        if fold in CFG.fold:
            # without leak tta for val_idx
            normal_val_idx = val_idx[val_idx<(5000000//CFG.seq_len)]
            tta_val_idx = val_idx[val_idx>=(5000000//CFG.seq_len)]
            no_leak_tta_val_idx = [i for i in tta_val_idx if (i-1000)%99%2==0]
            val_idx = list(tta_val_idx) + list(no_leak_tta_val_idx)
            print(f'len(val_idx): {len(val_idx)}')
            with timer(f'##### Running Fold: {fold} #####'):
                _predictions, _groundtruth = run(fold, trn_idx, val_idx)
                predictions.append(_predictions)
                groundtruth.append(_groundtruth)
    predictions = np.concatenate(predictions)
    groundtruth = np.concatenate(groundtruth)
    score = f1_score(predictions, groundtruth, labels=list(range(11)), average='macro')
    logger.info(f'##### CV Score: {score} #####')


if __name__ == '__main__':
    folds = pd.read_csv('../input/ion-prepare-data/folds.csv')
    main(X_train, folds)

