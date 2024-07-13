import librosa
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
import random
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import torchmetrics
import os
import warnings
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import copy

warnings.filterwarnings('ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Configuration 설정 클래스
class Config:
    SR = 32000
    N_MFCC = 13
    ROOT_FOLDER = './'
    N_CLASSES = 2
    BATCH_SIZE = 96
    N_EPOCHS = 15
    LR = 3e-4
    SEED = 42


CONFIG = Config()


# 랜덤 시드 설정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CONFIG.SEED)

# 데이터 로드
df = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'train.csv'))


# MFCC 특징 추출 함수
def get_mfcc_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows(), desc="MFCC Features Extraction", unit="files", total=len(df)):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
        mfcc = np.mean(mfcc.T, axis=0)
        features.append(mfcc)
        if train_mode:
            label = row['label']
            label_index = 0 if label == 'fake' else 1
            labels.append(label_index)
    return (features, labels) if train_mode else features


def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)

    # sns.lineplot에 키워드 인수로 전달
    sns.lineplot(x=epochs, y=train_losses, label='Training Loss', marker='o')
    sns.lineplot(x=epochs, y=val_losses, label='Validation Loss', marker='o')

    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# Custom Dataset 클래스
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


# 훈련, 검증 데이터 분할 및 로더 생성
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)
train_mfcc, train_labels = get_mfcc_feature(train, True)
val_mfcc, val_labels = get_mfcc_feature(val, True)

train_dataset = CustomDataset(train_mfcc, train_labels)
val_dataset = CustomDataset(val_mfcc, val_labels)

train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)


# MLP 모델 클래스
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
        return self.fc3(x)  # 최종 로짓 반환


# CNN 모델 클래스
class CNN(nn.Module):
    def __init__(self, input_dim=CONFIG.N_MFCC, hidden_dim=128, output_dim=CONFIG.N_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim // 2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=1, padding=1)
        conv_output_size = input_dim // 2 // 2  # 두 개의 풀링 레이어 후의 출력 크기 계산
        self.fc1 = nn.Linear(hidden_dim * conv_output_size, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # 채널 차원 추가
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 텐서 평탄화
        return self.fc1(x)


# RNN 모델 클래스
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 입력 텐서의 크기를 (batch_size, sequence_length, input_size)로 맞춤
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(
            x.device)  # (num_layers*2, batch_size, hidden_dim)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(
            x.device)  # (num_layers*2, batch_size, hidden_dim)

        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])  # 마지막 시퀀스의 출력값을 사용


# 모델 훈련 함수
def train(model, optimizer, train_loader, val_loader, device, n_epochs):
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_loss = float('inf')
    best_model = None
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)  # labels should be (batch_size,) with class indices
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device, dtype=torch.long)
                output = model(features)
                loss = criterion(output, labels)

                val_loss += loss.item()

                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = copy.deepcopy(model)

    plot_learning_curves(train_losses, val_losses)
    return best_model


# 다중 레이블 AUC 계산 함수
def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score


# 검증 함수
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_reality = []
    val_predictions = []

    tqdm_val = tqdm(val_loader, desc="Validation")
    for features, labels in tqdm_val:
        features = features.float().to(device)
        labels = labels.to(device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(features)
        loss = criterion(outputs, labels)
        val_loss.append(loss.item())
        val_reality.extend(labels.cpu().numpy())
        val_predictions.extend(F.softmax(outputs, dim=1).cpu().numpy())  # 수정된 부분
        tqdm_val.set_postfix(validation_loss=np.mean(val_loss))

    auc_score = roc_auc_score(val_reality, val_predictions, multi_class="ovr", average="macro")
    tqdm_val.close()
    return np.mean(val_loss), auc_score


# K-fold 교차 검증 함수
def kfold_cross_validation(model_class, data, labels, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=CONFIG.SEED)
    models = []
    for train_index, val_index in kf.split(data):
        train_data, val_data = [data[i] for i in train_index], [data[i] for i in val_index]
        train_labels, val_labels = [labels[i] for i in train_index], [labels[i] for i in val_index]
        train_dataset = CustomDataset(train_data, train_labels)
        val_dataset = CustomDataset(val_data, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

        if model_class == RNNModel:
            model = model_class(input_dim=CONFIG.N_MFCC, hidden_dim=128, num_layers=2, output_dim=CONFIG.N_CLASSES)
        else:
            model = model_class()

        optimizer = torch.optim.Adam(params=model.parameters(), lr=CONFIG.LR)
        # 정확한 매개변수를 제공하며 train 함수 호출
        best_model = train(model, optimizer, train_loader, val_loader, device, CONFIG.N_EPOCHS)
        models.append(best_model)
    return models


# 앙상블 예측 함수
def ensemble_inference(models, test_loader, device):
    for model in models:
        model.to(device)
    predictions = []
    for model in models:
        model.eval()
        model_preds = inference(model, test_loader, device)
        predictions.append(model_preds)
    ensemble_preds = np.mean(predictions, axis=0)
    return ensemble_preds


# 단일 모델 예측 함수
def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(test_loader, desc="Inferring", unit="batch", total=len(test_loader),
                             bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'):
            features = features.float().to(device)
            probs = model(features)
            probs = F.softmax(probs, dim=1).cpu().numpy()  # 수정된 부분
            predictions += probs.tolist()
    return predictions


# 테스트 데이터 로드 및 특징 추출
test = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'test.csv'))
test_mfcc = get_mfcc_feature(test, False)
test_dataset = CustomDataset(test_mfcc, None)

test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

# 모델 클래스들
models = [MLP, CNN, RNNModel]
trained_models = {}

# 모델 훈련
for model_class in models:
    trained_models[model_class.__name__] = kfold_cross_validation(model_class, train_mfcc, train_labels)


# 앙상블 예측 결과 평균 함수
def average_models(models_dict, test_loader, device):
    final_preds = None
    cnt = 0
    for model_name, models in models_dict.items():
        model_preds = ensemble_inference(models, test_loader, device)
        if final_preds is None:
            final_preds = model_preds
        else:
            final_preds += model_preds
        cnt += 1
    return final_preds / cnt


# 앙상블 예측 진행 및 결과 저장
ensemble_preds = average_models(trained_models, test_loader, device)
submit = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'sample_submission.csv'))
submit.iloc[:, 1:] = ensemble_preds
submit.to_csv('./ensemble3_submit.csv', index=False)
