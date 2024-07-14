import librosa
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings('ignore')

# 설정 클래스
class Config:
    SR = 32000  # 샘플링 레이트
    N_MFCC = 13  # MFCC 계수 개수
    ROOT_FOLDER = './'
    N_CLASSES = 2  # 클래스 개수 (진짜, 가짜)
    N_ESTIMATORS = 50  # AdaBoost에 사용할 약한 학습기의 수
    SEED = 42  # 랜덤 시드

CONFIG = Config()

# 시드 고정 함수
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CONFIG.SEED) # 시드 고정

# 데이터셋 로드
df = pd.read_csv('train.csv')
train, val = train_test_split(df, test_size=0.2, random_state=CONFIG.SEED)

# MFCC 특징 추출 함수
def get_mfcc_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
        mfcc = np.mean(mfcc.T, axis=0)
        features.append(mfcc)
        if train_mode:
            labels.append(row['label'])
    return features, labels

train_mfcc, train_labels = get_mfcc_feature(train, True)
val_mfcc, val_labels = get_mfcc_feature(val, True)

# 라벨을 이진화
train_labels = [0 if label == 'fake' else 1 for label in train_labels]
val_labels = [0 if label == 'fake' else 1 for label in val_labels]

# 데이터 불균형 문제 해결 - SMOTE 사용
sm = SMOTE(random_state=CONFIG.SEED)
train_mfcc_resampled, train_labels_resampled = sm.fit_resample(train_mfcc, train_labels)

# AdaBoost 분류기 초기화 및 학습
# 기본 학습기로 깊이가 더 깊은 결정 트리를 사용
base_estimator = DecisionTreeClassifier(max_depth=3, random_state=CONFIG.SEED)
model = AdaBoostClassifier(estimator=base_estimator, n_estimators=CONFIG.N_ESTIMATORS, random_state=CONFIG.SEED)
model.fit(train_mfcc_resampled, train_labels_resampled)

# 모델 검증
val_predictions = model.predict_proba(val_mfcc)
val_auc = roc_auc_score(val_labels, val_predictions[:, 1])
print(f'Validation AUC: {val_auc:.5f}')

# 테스트 데이터셋 로드 및 MFCC 특징 추출
test = pd.read_csv('test.csv')
test_mfcc, _ = get_mfcc_feature(test, False)

# 테스트 데이터 예측
test_predictions = model.predict_proba(test_mfcc)

# 제출 파일 생성
submit = pd.read_csv('sample_submission.csv')
submit.iloc[:, 1:] = test_predictions
submit.to_csv('adaboost_submit.csv', index=False)
