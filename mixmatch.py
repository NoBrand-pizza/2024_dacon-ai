import librosa
import numpy as np
import pandas as pd
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import os

warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Config:
    SR = 32000
    N_MFCC = 13
    ROOT_FOLDER = './'
    N_CLASSES = 2
    BATCH_SIZE = 96
    N_EPOCHS = 5
    LR = 3e-4
    SEED = 42

CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED)

df = pd.read_csv('train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)

def get_mfcc_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
        mfcc = np.mean(mfcc.T, axis=0)
        features.append(mfcc)
        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)
    if train_mode:
        return features, labels
    return features

train_mfcc, train_labels = get_mfcc_feature(train, True)
val_mfcc, val_labels = get_mfcc_feature(val, True)

class CustomDataset(Dataset):
    def __init__(self, mfcc, label):
        self.mfcc = mfcc
        self.label = label
    def __len__(self):
        return len(self.mfcc)
    def __getitem__(self, index):
        if self.label is not None:
            return self.mfcc[index], self.label[index]
        return self.mfcc[index]

train_dataset = CustomDataset(train_mfcc, train_labels)
val_dataset = CustomDataset(val_mfcc, val_labels)

train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_dim=CONFIG.N_MFCC, hidden_dim=128, output_dim=CONFIG.N_CLASSES):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

def mixup_data(x, y, alpha=1.0):
    '''MixUp 데이터 증강 기법 적용'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''MixUp 손실 계산'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def sharpen(p, T=0.5):
    '''예측 값을 균일화 (Sharpening)'''
    p = p ** (1 / T)
    return p / p.sum(dim=1, keepdim=True)

def mixmatch(model, x_l, y_l, x_ul, alpha=0.75, T=0.5, K=2):
    '''MixMatch 알고리즘 적용'''
    # Unlabeled 데이터에 대한 K개의 예측 평균 및 Sharpening 적용
    with torch.no_grad():
        logits = torch.zeros(K, x_ul.size(0), CONFIG.N_CLASSES).to(device)
        for k in range(K):
            logits[k] = model(x_ul)
        q_bar = torch.mean(F.softmax(logits, dim=2), dim=0)
        q = sharpen(q_bar, T)

    # Labeled 데이터와 Unlabeled 데이터를 혼합
    x = torch.cat([x_l, x_ul], dim=0)
    y = torch.cat([y_l, q], dim=0)
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha)

    # Labeled 데이터와 Unlabeled 데이터로 나누기
    mixed_x_l = mixed_x[:x_l.size(0)]
    mixed_x_ul = mixed_x[x_l.size(0):]
    y_l_a, y_l_b = y_a[:x_l.size(0)], y_b[:x_l.size(0)]
    y_ul_a, y_ul_b = y_a[x_l.size(0):], y_b[x_l.size(0):]

    return mixed_x_l, y_l_a, y_l_b, mixed_x_ul, y_ul_a, y_ul_b, lam

def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    best_val_score = 0
    best_model = None

    for epoch in range(1, CONFIG.N_EPOCHS + 1):
        model.train()
        train_loss = []
        for batch in tqdm(train_loader):
            features, labels = batch
            features = features.float().to(device)
            labels = labels.float().to(device)

            # MixMatch 적용
            mixed_x_l, y_l_a, y_l_b, mixed_x_ul, y_ul_a, y_ul_b, lam = mixmatch(
                model, features, labels, features, alpha=0.75, T=0.5, K=2)

            optimizer.zero_grad()
            output_l = model(mixed_x_l)
            output_ul = model(mixed_x_ul)

            # Labeled 데이터와 Unlabeled 데이터에 대한 손실 계산
            loss_l = mixup_criterion(criterion, output_l, y_l_a, y_l_b, lam)
            loss_ul = mixup_criterion(criterion, output_ul, y_ul_a, y_ul_b, lam)
            loss = loss_l + loss_ul

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')

        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model

    return best_model

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            probs = model(features)
            loss = criterion(probs, labels)
            val_loss.append(loss.item())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        _val_loss = np.mean(val_loss)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        auc_score = multiLabel_AUC(all_labels, all_probs)
    return _val_loss, auc_score

model = MLP()
optimizer = torch.optim.Adam(params=model.parameters(), lr=CONFIG.LR)
infer_model = train(model, optimizer, train_loader, val_loader, device)

test = pd.read_csv('./test.csv')
test_mfcc = get_mfcc_feature(test, False)
test_dataset = CustomDataset(test_mfcc, None)
test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)
            probs = model(features)
            probs = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions

preds = inference(infer_model, test_loader, device)
submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:, 1:] = preds
submit.head()
submit.to_csv('./mixmatch_submit.csv', index=False)