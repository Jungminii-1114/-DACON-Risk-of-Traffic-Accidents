#!/usr/bin/env python
# coding: utf-8

# ## Submission format
# - Test_id : test identication ID
# - Label : Predicted probability of accident Risk class (0~1)
# 
# ## data/train.csv (944,767 rows)
# - Test_id : test identication ID
# - Test : Class of test (A : Qualification test for new, B : Qualification for maintain)
# 
# 
# ## data/train/A.csv : Qualicifation test for 'new' of train datasets (647,241 rows)
# - Test_id : test identication ID
# - Test : Class of test (A : Qualification test for new, B : Qualification for maintain)
# - PrimaryKey : ID for driver
# - Age : Age for driver
# - TestDate : Date of test
# 
# ## data/train/B.csv : Qualification test for 'maintain' of train datasets (297,526 rows)
# - Test_id : test identication ID
# - Test : Class of test (A : Qualification test for new, B : Qualification for maintain)
# - PrimaryKey : ID for driver
# - Age : Age for driver
# - TestDate : Date of test
# 
# 
# ## A.csv + B.csv = Whole Train datasets

# In[1]:


# import pkg_resources
# import re

# # 코드 전체 읽기
# with open("your_notebook.ipynb", encoding="utf-8") as f:
#     text = f.read()

# # import된 패키지 이름 추출 (정규식)
# imports = set(re.findall(r'(?<=import )\w+|(?<=from )\w+', text))

# # 현재 환경에 설치된 패키지 목록
# installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

# # requirements.txt로 저장
# with open("requirements.txt", "w") as f:
#     for name in sorted(imports):
#         if name.lower() in installed:
#             f.write(f"{name}=={installed[name.lower()]}\n")
# print("requirements.txt 생성 완료!")


# In[2]:


import subprocess

# subprocess.run(["pip", "install", "-r", "requirements.txt"])


# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
#%matplotlib inline

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss


# In[4]:


train_main = pd.read_csv('data/train.csv')
test_main = pd.read_csv('data/test.csv')

train_A = pd.read_csv('data/train/A.csv')
train_B = pd.read_csv('data/train/B.csv')
test_A = pd.read_csv('data/test/A.csv')
test_B = pd.read_csv('data/test/B.csv')


# In[5]:


train_A_full = pd.merge(train_main[train_main['Test'] == 'A'], train_A, on='Test_id', how='left')
test_A_full = pd.merge(test_main[test_main['Test'] == 'A'], test_A, on='Test_id', how='left')

train_B_full = pd.merge(train_main[train_main['Test'] == 'B'], train_B, on='Test_id', how='left')
test_B_full = pd.merge(test_main[test_main['Test'] == 'B'], test_B, on='Test_id', how='left')


# # 1. Dealing with 'train_A_full'

# ## 1-1. Preprocessing (train_A_full)

# In[6]:


train_A_full.info()


# In[7]:


train_A_full.head(5)


# In[8]:


train_A_full = train_A_full.drop('PrimaryKey', axis=1)
train_A_full = train_A_full.drop('Test_id', axis=1)


# In[9]:


train_A_full.head(5)


# ## 1-2. Basic data information
# 

# In[10]:


# img1 = cv2.imread("data_info/class_A1.png")
# img2 = cv2.imread('data_info/class_A2.png')
# img3 = cv2.imread('data_info/class_A3.png')
# img4 = cv2.imread("data_info/class_A4.png")

# fig = plt.figure(figsize=(12,4))
# rows=1
# cols=2

# ax1 = fig.add_subplot(rows, cols, 1)
# ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
# ax1.set_title("test_A1 info")
# ax1.axis('off')

# ax2 = fig.add_subplot(rows, cols, 2)
# ax2.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
# ax2.set_title("test_A2 info")
# ax2.axis('off')

# plt.show()


# ### 결측치 확인하기 

# In[11]:


train_A_full.isnull().sum()


# In[12]:


train_A_full[train_A_full.isna().any(axis=1)]


# In[13]:


train_A_full[train_A_full['A5-3'].isnull()]


# A2 계열의 결측치 : 31210번 인덱스의 데이터 
# A3 계열의 결측치 : 19390번 인덱스의 데이터 
# A5 계열의 결측치 : 40404, 628227 인덱스의 데이터

# ## 1-3. The way 01 : 완전삭제법
# 

# ## 아직 미반영 : 
# 
# ### 1. 주최측의 경우 A3-6 --> Trials(33-36)에 대한 내용은 Dummy Data로 간주하여도 문제 없다고 공지. (25.10.16. 09:30)
# 
# 
# ### 2. A2-1, A2-2 항목의 데이터 수집 과정에서 일부 Dummy 정보 포함되었을 가능성 있음
# 
# 

# In[14]:


train_A_full = train_A_full.drop([31210, 19390, 40404, 628227], axis=0)


# In[15]:


train_A_full.describe()


# In[16]:


train_A_full.isnull().sum()


# ## Age Column을 나이대별 초반/중반으로 나눠서 2, 7로 클래스 나누기

# In[17]:


# 예시: '30a' -> 30, 'a' -> 0 / 'b' -> 5로 변환 후 더하기
def process_age(age_str):
    base_age = int(age_str[:-1]) # '30a' -> 30
    suffix = age_str[-1]         # '30a' -> 'a'
    
    if suffix == 'a':
        return base_age + 2 # (0~4세의 중간값 2)
    elif suffix == 'b':
        return base_age + 7 # (5~9세의 중간값 7)
    return base_age # 예외 처리

# 'Age' 컬럼에 적용
train_A_full['Age_numeric'] = train_A_full['Age'].apply(process_age)


# In[18]:


train_A_full.head(3)


# ### Age_numeric을 새로 만들었으니 기존 Age Column은 삭제

# In[19]:


train_A_full = train_A_full.drop('Age', axis=1)


# In[20]:


train_A_full.head(3)


# ## A1-3 Column : 1의 개수 나타내는 column 생성 

# In[21]:


def parse_sequence(seq_str):
    try:
        # 콤마(,)로 분리하고, 각 항목을 float(실수)으로 변환
        return [float(x) for x in seq_str.split(',')]
    except:
        # 혹시 모를 에러 (빈 값 등)가 나면 빈 리스트 반환
        return []


# In[22]:


train_A_full['A1-3_num_of_incorrect'] = train_A_full['A1-3'].str.count('1')


# In[23]:


train_A_full.head(5)


# In[24]:


column_to_drop = ['A1-1', 'A1-2', 'A1-3', 'A1-4']
train_A_full = train_A_full.drop(column_to_drop, axis=1)


# ## A8 ~ A9 : Drop ==> 이걸 유의미한 데이터로 뽑기에는 아직 감이 안잡힘 

# In[25]:


# 'Age' 컬럼의 인덱스 번호가 궁금할 때
idx = train_A_full.columns.get_loc('A9-5')
print(f"'Age' 컬럼의 인덱스 번호: {idx}")

# 31 ~ 35


# In[26]:


cols_to_drop = ['A8-1', 'A8-2', 'A9-1', 'A9-2', 'A9-3', 'A9-4', 'A9-5']
train_A_full = train_A_full.drop(cols_to_drop, axis=1)


# In[27]:


train_A_full


# ## A7 구간화 하기 

# In[28]:


train_A_full['A7-1'].describe()


# In[29]:


bins_A7 = [-1, 7, 13, 18]

labels=['Low', 'Average', 'High']
train_A_full['A7-1'] = pd.cut(train_A_full['A7-1'], bins=bins_A7, labels=labels, right=True)


# In[30]:


train_A_full.head(5)


# ## A6 구간화 하기 

# In[31]:


# -1로 시작해서 0 포함하기 
# (0-7), (8-12), (13-14)
bins = [-1, 7, 12, 14]

labels = ['Low', 'Average', 'High']
train_A_full['A6-1'] = pd.cut(train_A_full['A6-1'], bins=bins, labels=labels, right=True)


# In[32]:


train_A_full.head(5)


# ## A5

# In[33]:


cols_to_drop = ['A5-1', 'A5-2', 'A5-3']

train_A_full['A5-3_num_of_incorrect'] = train_A_full['A5-3'].str.count('1')
# train_A_full = train_A_full.drop(cols_to_drop)


# In[34]:


train_A_full = train_A_full.drop(cols_to_drop, axis=1)
train_A_full


# ## A4

# In[35]:


train_A_full['A4-5'].describe()


# In[36]:


labels=['Very_Fast', 'Fast', 'Slow', 'Very_Slow']

train_A_full['A4-5'] = train_A_full['A4-5'].apply(parse_sequence)
train_A_full['A4-5'] = train_A_full['A4-5'].apply(np.sum)

train_A_full['A4-5'].describe()


# In[37]:


bins=[-1, 45799, 49540, 54464, 240166]
train_A_full['A4-5']=pd.cut(train_A_full['A4-5'], bins=bins, labels=labels, right=True, duplicates='drop')
train_A_full['A4-5']


# In[38]:


cols_to_drop = ['A4-1', 'A4-2', 'A4-3', 'A4-4']
train_A_full = train_A_full.drop(cols_to_drop, axis=1)
train_A_full


# ## A3
# 
# - 이건 반응 속도가 중요한 test case.
# 
# - 다 맞추더라도 시간이 늦으면 점수가 낮음 

# In[39]:


train_A_full['A3-5'] = train_A_full['A3-5'].apply(parse_sequence)
train_A_full['A3-7'] = train_A_full['A3-7'].apply(parse_sequence)


# In[40]:


INCORRECT_PENALTY = 5000
def calculate_a3_penalty(correctness_list, time_list):
    total_penalty = 0
    for correct_code, time in zip(correctness_list, time_list):
        if correct_code == 1:
            total_penalty += time
        else:
            total_penalty += INCORRECT_PENALTY
    return total_penalty

train_A_full['A3_tot_penalty'] = train_A_full.apply(lambda row : calculate_a3_penalty(row['A3-5'], row['A3-7']), axis=1)
train_A_full[['A3_tot_penalty', 'Label']].head()


# In[41]:


cols_to_drop = ['A3-1', 'A3-2', 'A3-3', 'A3-4', 'A3-5', 'A3-6', 'A3-7']
train_A_full = train_A_full.drop(cols_to_drop, axis=1)
train_A_full


# In[42]:


train_A_full['A3_tot_penalty'].describe()


# In[43]:


bins=[50186, 56454, 59206, 66954, 160000]
labels = ['Best', 'God', 'Average', 'Worst']
train_A_full['A3_Penalty_Group'] = pd.cut(train_A_full['A3_tot_penalty'], bins=bins, labels=labels, include_lowest=True)
train_A_full['A3_Penalty_Group'].value_counts()


# ## A2

# In[44]:


train_A_full.columns


# In[45]:


# A2-3 : 정답 유무
# A2-4 : 시간  --> 이것도 내 생각에는 0에 가까울 수록 좋은것 같은데.. 아직 처리하기에 너무 어려우니까 drop..

train_A_full['A2-3_num_of_incorrect'] = train_A_full['A2-3'].str.count('1')


# In[46]:


cols_to_drop = ['A2-1', 'A2-2', 'A2-4']
train_A_full = train_A_full.drop(cols_to_drop, axis=1)


# In[47]:


train_A_full.head(4)


# In[48]:


train_A_full = train_A_full.drop('A2-3', axis=1)
train_A_full


# ## Delete TestDate and Test_x

# In[49]:


train_A_full = train_A_full.drop(['TestDate', 'Test_x'], axis=1)
train_A_full


# ## 필요한 컬럼만 사용하기

# In[50]:


features_2use = [
    'Age_numeric',
    'A1-3_num_of_incorrect',
    'A5-3_num_of_incorrect',
    'A3_tot_penalty',
    'A3_Penalty_Group',
    'A4-5',
    'A6-1',
    'A7-1'
]

X = train_A_full[features_2use]
y = train_A_full['Label']


# # Encoding Categorical Feature

# In[51]:


categorical_features = ['A3_Penalty_Group', 'A4-5', 'A6-1', 'A7-1']
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
X_encoded.head(20)


# # train_test_split

# In[52]:


X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
print(f"train data : {X_train.shape}, {y_train.shape}")
print(f"Validation data : {X_val.shape}, {y_val.shape}")


# In[53]:


model_A = LGBMClassifier(random_state=42, class_weight='balanced')
print("Start Training Model A . . . ")
model_A.fit(X_train, y_train)
print("Train finished!!")


# In[54]:


preds_val = model_A.predict(X_val)
proba_val = model_A.predict_proba(X_val)[:,1] # 1이 될 확률

acc=accuracy_score(y_val, preds_val)
auc = roc_auc_score(y_val, proba_val)

print(f"--- Validation set Grade ---")
print(f"Accuracy : {acc:.4f}")
print(f"AUC Score : {auc:.4f}")


# # Feature Importance or correlation

# In[55]:


feature_names = X_encoded.columns

importances = model_A.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 12))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Model A - Feature Importance')
plt.show()

print(feature_importance_df.head(10))


# ## Hypter Parameter setting 

# ## Method 1 : Using Optuna

# In[56]:


#!pip install optuna


# In[57]:


import optuna

# 1. 평가 함수 정의 (Optuna가 이 함수를 반복 호출합니다)
def objective(trial):
    # Optuna가 테스트할 하이퍼파라미터 값의 범위를 지정합니다.
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1, # 로그(출력)를 최소화합니다.
        'boosting_type': 'gbdt',
        'class_weight': 'balanced', # 불균형 데이터 처리 (필수)
        'random_state': 42,
        
        # --- 튜닝 대상 파라미터 ---
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), # 데이터 샘플링 비율
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0) # 피처 샘플링 비율
    }
    
    # 1. 모델 정의
    model = LGBMClassifier(**params)
    
    # 2. 모델 학습
    model.fit(X_train, y_train)
    
    # 3. 검증(Validation) 세트로 AUC 점수 평가
    proba_val = model.predict_proba(X_val)[:, 1]
#    auc = roc_auc_score(y_val, proba_val)
    brier_loss = brier_score_loss(y_val, proba_val)
    
    return brier_loss # Optuna는 이 AUC 점수를 '최대화'하는 방향으로 파라미터를 찾습니다.

# 3. 튜닝 시작
# direction='maximize': objective 함수가 반환하는 값(AUC)을 최대화합니다.
study = optuna.create_study(direction='minimize')

# 4. 튜닝 실행 (n_trials=30 : 30번의 각기 다른 파라미터 조합으로 시도)
# (시간이 있다면 50~100회로 늘리면 더 좋습니다)
print("Optuna 튜닝을 시작합니다... (약 5~10분 소요될 수 있습니다)")
study.optimize(objective, n_trials=30) 

# 5. 튜닝 결과 확인
print("--- 튜닝 완료! ---")
print(f"찾아낸 최적의 Brier 점수: {study.best_value:.4f}")
print("최적의 하이퍼파라미터:")
print(study.best_params)


# In[58]:


# 1. Optuna가 찾아낸 최적의 파라미터 가져오기
best_params = study.best_params

# 2. class_weight와 random_state를 기본값으로 추가
best_params['class_weight'] = 'balanced'
best_params['random_state'] = 42

# 3. 최종 모델(Final Model A) 학습
final_model_A = LGBMClassifier(**best_params)

print("\n최적의 파라미터로 최종 모델 A를 학습합니다...")
final_model_A.fit(X_train, y_train)

# 4. 최종 모델 성능 검증
proba_val_final = final_model_A.predict_proba(X_val)[:, 1]
final_auc = roc_auc_score(y_val, proba_val_final)

print(f"--- 최종 모델 성능 (검증 세트) ---")
print(f"기본 모델 AUC: 0.6479")
print(f"튜닝된 모델 AUC: {final_auc:.4f}")


# ## 점수가 낮은 관계로.. 다시 Feature Engineering..

# In[59]:


cols_to_drop = ['A3_Penalty_Group_God', 'A3_Penalty_Group_Average', 'A3_Penalty_Group_Worst']
cols_to_drop_safe = [col for col in cols_to_drop if col in X_encoded.columns]
X_encoded_v2 = X_encoded.drop(columns=cols_to_drop_safe)


# In[60]:


X_train, X_val, y_train, y_val = train_test_split(X_encoded_v2, y, test_size=0.2, random_state=42, stratify=y)
print(f"train data : {X_train.shape}, {y_train.shape}")
print(f"Validation data : {X_val.shape}, {y_val.shape}")


# In[61]:


model_A_v2 = LGBMClassifier(random_state=42, class_weight='balanced')
print("Start Training Model A . . . ")
model_A_v2.fit(X_train, y_train)
print("Train finished!!")


# In[62]:


preds_val = model_A_v2.predict(X_val)
proba_val = model_A_v2.predict_proba(X_val)[:,1] # 1이 될 확률

acc=accuracy_score(y_val, preds_val)
auc = roc_auc_score(y_val, proba_val)

print(f"--- Validation set Grade ---")
print(f"Accuracy : {acc:.4f}")
print(f"AUC Score : {auc:.4f}")


# # Start with B

# In[63]:


train_B_full.info()


# In[64]:


train_B_full.head()


# In[65]:


train_B_full = train_B_full.drop(['Test_id', 'PrimaryKey', 'TestDate'], axis=1)
train_B_full.head()


# In[66]:


print("--- B6-1 (응답 후보) 고유값 분포 ---")
try:
    # B6-1 컬럼을 리스트로 변환
    B6_list = train_B_full['B6-1'].apply(parse_sequence)
    # 리스트들을 한 줄로 세워서(stack) 고유값 개수 확인
    unique_values_B6 = B6_list.apply(pd.Series).stack().value_counts()
    print(unique_values_B6)
except Exception as e:
    print(f"B6-1 처리 중 오류: {e}")


print("\n--- B9-1 (0/1 후보) 고유값 분포 ---")
try:
    # B9-1 컬럼을 리스트로 변환
    B9_list = train_B_full['B9-1'].apply(parse_sequence)
    # 리스트들을 한 줄로 세워서(stack) 고유값 개수 확인
    unique_values_B9 = B9_list.apply(pd.Series).stack().value_counts()
    print(unique_values_B9)
except Exception as e:
    print(f"B9-1 처리 중 오류: {e}")


# # 교차검증을 통한 모델 선택 

# In[67]:


# classification_models={
#     "Random Forest Classifier" : RandomForestClassifier(),
#     "Gradient Boosting Classifier" : GradientBoostingClassifier(),
#     "Logistic Regression" : LogisticRegression(max_iter = 1000),
#     "Decision Tree Classifier" : DecisionTreeClassifier(),
#     "SGD Classifier" : SGDClassifier(),
#     "KNN Classifier" : KNeighborsClassifier(n_neighbors=5)
# }


# In[68]:


# cv_scores_kfold = {}
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # cross_val_score() 사용하지 않고 kf.split을 사용하여 수동 루프 구동
# for name, model in classification_models.items():
#     cv_accuracy_kfold=[]
#     n_iter=0
#     print(f"\n [{name}]")
    
#     for train_idx, val_idx in kf.split(X_train, y_train): # X 전체가 아닌 훈련 셋에 대해 교차검증
#         X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
#         # 학습 
#         model.fit(X_train_fold, y_train_fold)
        
#         pred = model.predict(X_val_fold)
#         acc = accuracy_score(y_val_fold, pred)
#         cv_accuracy_kfold.append(acc)
        
#         print(f"Fold {n_iter+1} 정확도 : {acc:.4f}")
#         n_iter += 1
#     print(f"-> 평균 졍확도 : {np.mean(cv_accuracy_kfold):.4f}, 표준편차 : {np.std(cv_accuracy_kfold):.4f}")
              


# # Submission of Model_A

# In[70]:


import pandas as pd
import numpy as np
import optuna
import os
import joblib
import json

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier

DATA_DIR = './data'
#DATA_DIR = './open'  # DACON 조건
OUTPUT_DIR = './output'
#OUTPUT_DIR = '.'  # DACON 조건
MODEL_DIR = './model'

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True )
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, 'submission.csv')

A_MODEL_PATH = os.path.join(MODEL_DIR, "model_A.joblib")
B_MODEL_PATH = os.path.join(MODEL_DIR, "model_B.joblib")
A_PREPROC_PATH = os.path.join(MODEL_DIR, "preproc_A.joblib")
B_PREPROC_PATH = os.path.join(MODEL_DIR, "preproc_B.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.json")


# --- 2. 헬퍼 함수 정의 (재사용) ---

def parse_sequence(seq_str):
    """
    '1,2,3,4' 같은 텍스트 시퀀스를 [1.0, 2.0, 3.0, 4.0] 리스트로 변환합니다.
    """
    try:
        return [float(x) for x in str(seq_str).split(',')]
    except:
        return []

def process_age(age_str):
    """
    '30a', '60b' 같은 나이 형식을 숫자로 변환합니다. (예: 32, 67)
    """
    try:
        base_age = int(str(age_str)[:-1])
        suffix = str(age_str)[-1]
        
        if suffix == 'a':
            return base_age + 2 # (0~4세의 중간값 2)
        elif suffix == 'b':
            return base_age + 7 # (5~9세의 중간값 7)
    except:
        return np.nan # 변환 실패 시 결측치
    return np.nan

# A3 페널티 계산 함수 (A-데이터 전용)
INCORRECT_PENALTY = 5000 # 오답 시 부여할 고정 페널티 (ms)

def calculate_a3_penalty(correctness_list, time_list):
    """
    A3-5 (정답)와 A3-7 (시간)을 조합하여 총 페널티 점수를 계산합니다.
    """
    total_penalty = 0
    
    # 두 리스트 길이가 다르면 0점 반환 (예외 처리)
    if len(correctness_list) != len(time_list):
        return 0

    for correct_code, time in zip(correctness_list, time_list):
        if correct_code == 1: # 1: 'valid correct' (정답)
            total_penalty += time
        else: # 2, 3, 4 (오답)
            total_penalty += INCORRECT_PENALTY
            
    return total_penalty

# --- 3. 피처 엔지니어링 함수 ---

def feature_engineer_A(df):
    """
    A-데이터셋 (신규 검사자)에 특화된 피처 엔지니어링을 수행합니다.
    """
    print("A-데이터 피처 엔지니어링 시작...")
    
    # 1. Age (나이)
    df['Age_numeric'] = df['Age'].apply(process_age)
    
    # 2. TestDate (검사일)
    df['TestDate_str'] = df['TestDate'].astype(str)
    df['Test_Year'] = df['TestDate_str'].str[:4].astype(int)
    df['Test_Month'] = df['TestDate_str'].str[4:].astype(int)

    # 3. 부정확한 응답 개수 (Penalty)
    df['A1-3_num_of_incorrect'] = df['A1-3'].str.count('1')
    df['A5-3_num_of_incorrect'] = df['A5-3'].str.count('1')
    df['A2-3_num_of_incorrect'] = df['A2-3'].str.count('1')

    # 4. A4-5 (시험 시간) 구간화
    df['A4-5_list'] = df['A4-5'].apply(parse_sequence)
    df['A4-5_sum'] = df['A4-5_list'].apply(np.sum)
    labels_a4 = ['Very_Fast', 'Fast', 'Slow', 'Very_Slow']
    # qcut(개수 기준)으로 4그룹 분리
    df['A4-5_Group'] = pd.qcut(df['A4-5_sum'], q=4, labels=labels_a4, duplicates='drop')

    # 5. A6-1 (정수값) 구간화 (표준편차 기반 3그룹)
    bins_a6 = [-1, 7, 12, 14]
    labels_a6 = ['A6_Low', 'A6_Average', 'A6_High']
    df['A6-1_Group'] = pd.cut(df['A6-1'], bins=bins_a6, labels=labels_a6)

    # 6. A7-1 (시간) 구간화 (qcut 3그룹)
    df['A7-1_list'] = df['A7-1'].apply(parse_sequence)
    df['A7-1_sum'] = df['A7-1_list'].apply(np.sum)
    labels_a7 = ['A7_Low', 'A7_Average', 'A7_High']
    df['A7-1_Group'] = pd.qcut(df['A7-1_sum'], q=3, labels=labels_a7, duplicates='drop')

    # 7. A3 Total Penalty (핵심 피처)
    df['A3-5_list'] = df['A3-5'].apply(parse_sequence)
    df['A3-7_list'] = df['A3-7'].apply(parse_sequence)
    df['A3_tot_penalty'] = df.apply(lambda row: calculate_a3_penalty(row['A3-5_list'], row['A3-7_list']), axis=1)
    
    # 8. 범주형 피처 원-핫 인코딩
    categorical_features = ['A4-5_Group', 'A6-1_Group', 'A7-1_Group']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # 9. 최종 사용할 피처 목록 정의
    # (원본 시퀀스 컬럼, 중간 계산 컬럼 등은 모두 제외)
    final_features_A = [
        'Age_numeric', 'Test_Year', 'Test_Month',
        'A1-3_num_of_incorrect', 'A5-3_num_of_incorrect', 'A2-3_num_of_incorrect',
        'A3_tot_penalty',
        'A4-5_Group_Fast', 'A4-5_Group_Slow', 'A4-5_Group_Very_Slow',
        'A6-1_Group_Average', 'A6-1_Group_High',
        'A7-1_Group_Average', 'A7-1_Group_High'
    ]
    
    # 피처가 존재하지 않을 경우를 대비하여, 있는 피처만 선택
    existing_features = [col for col in final_features_A if col in df.columns]
    
    return df[existing_features + ['Test_id']], existing_features
#    return df[existing_features + ['Test_id']] # Test_id는 나중에 병합을 위해 포함

def feature_engineer_B(df):
    """
    B-데이터셋 (자격 유지)에 특화된 피처 엔지니어링을 수행합니다.
    (수정됨: B6-1/B7-1 -> B6/B7로 컬럼명 수정, d1-d8 컬럼 삭제)
    """
    print("B-데이터 피처 엔지니어링 시작 (A-데이터 로직 적용)...")
    
    # 1. Age (나이)
    df['Age_numeric'] = df['Age'].apply(process_age)
    
    # 2. TestDate (검사일)
    df['TestDate_str'] = df['TestDate'].astype(str)
    df['Test_Year'] = df['TestDate_str'].str[:4].astype(int)
    df['Test_Month'] = df['TestDate_str'].str[4:].astype(int)

    # 3. 부정확한 응답 개수 (B9-1 사용)
    # (B9-1은 컬럼 리스트에 존재 확인)
    df['B9-1'] = df['B9-1'].astype(str) 
    df['B9_1_num_of_incorrect'] = df['B9-1'].str.count('1')

    # 4. B Total Penalty (B6, B7 사용)
    # (A3 페널티 계산 함수 calculate_a3_penalty를 재사용)
    # --- 수정됨: 'B6-1' -> 'B6', 'B7-1' -> 'B7' ---
    df['B6_list'] = df['B6'].apply(parse_sequence)
    df['B7_list'] = df['B7'].apply(parse_sequence)
    df['B_total_penalty'] = df.apply(
        # --- 수정됨: row['B6-1_list'] -> row['B6_list'] 등 ---
        lambda row: calculate_a3_penalty(row['B6_list'], row['B7_list']),
        axis=1 
    )
    
    # 5. d1 ~ d8 컬럼 삭제 (존재하지 않음)
    
    # 6. TODO: B-데이터만의 다른 시퀀스(B1-1, B2-1 등) 피처 추가
    
    # 7. 범주형 피처 (없으므로 통과)

    # 8. 최종 사용할 피처 목록 정의
    final_features_B = [
        'Age_numeric', 'Test_Year', 'Test_Month',
        'B9_1_num_of_incorrect',  # <-- B9-1은 이름이 맞았음
        'B_total_penalty',
        # d1 ~ d8 컬럼은 존재하지 않으므로 목록에서 제거
    ]
    
    # 피처가 존재하지 않을 경우를 대비하여, 있는 피처만 선택
    existing_features = [col for col in final_features_B if col in df.columns]

    return df[existing_features + ['Test_id']], existing_features

# --- 4. 메인 실행 함수 ---
def main():
    
    # (1) 데이터 로드
    print("--- 1. 데이터 로드 ---")
    try:
        train_main = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        test_main = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
        
        sample_submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
        
        train_A = pd.read_csv(os.path.join(DATA_DIR, 'train/A.csv'))
        train_B = pd.read_csv(os.path.join(DATA_DIR, 'train/B.csv'))
        test_A = pd.read_csv(os.path.join(DATA_DIR, 'test/A.csv'))
        test_B = pd.read_csv(os.path.join(DATA_DIR, 'test/B.csv'))
    except FileNotFoundError:
        print(f"오류: '{DATA_DIR}' 디렉터리에서 데이터를 찾을 수 없습니다.")
        print("빈 제출 파일을 생성하고 종료합니다.")
        dummy = pd.DataFrame({'Test_id': [], 'Label': []})
        dummy.to_csv(SUBMISSION_PATH, index=False)
        print(f"현재 DATA_DIR : {DATA_DIR}")
        return

    print("\n---[디버깅] train_B.csv 실제 컬럼 목록 ---")
    print(train_B.columns.tolist())
    
    print("\n---[디버깅] test_B.csv 실제 컬럼 목록 ---")
    print(test_B.columns.tolist())
    
    
    # (2) A-데이터 병합 및 처리
    print("\n--- 2. A-데이터 처리 ---")
    train_A_full = pd.merge(train_main[train_main['Test'] == 'A'], train_A, on='Test_id', how='left')
    test_A_full = pd.merge(test_main[test_main['Test'] == 'A'], test_A, on='Test_id', how='left')

    # A-데이터 피처 엔지니어링
    # (train, test를 합쳐서 전처리 후 다시 분리 - Dummies 일관성 유지)
    train_A_len = len(train_A_full)
    combined_A = pd.concat([train_A_full, test_A_full], ignore_index=True)
    
    processed_A, final_features_A_list = feature_engineer_A(combined_A)
    
    X_train_A = processed_A.iloc[:train_A_len].drop('Test_id', axis=1)
    X_test_A = processed_A.iloc[train_A_len:].drop('Test_id', axis=1)
    y_train_A = train_A_full['Label']
    test_A_ids = test_A_full['Test_id']

    print(f"A-전처리기(피처 목록) 저장 : {A_PREPROC_PATH}")
    joblib.dump(final_features_A_list, A_PREPROC_PATH)
    
    
    # (3) B-데이터 병합 및 처리
    print("\n--- 3. B-데이터 처리 ---")
    train_B_full = pd.merge(train_main[train_main['Test'] == 'B'], train_B, on='Test_id', how='left')
    test_B_full = pd.merge(test_main[test_main['Test'] == 'B'], test_B, on='Test_id', how='left')
    
    # B-데이터 피처 엔지니어링
    train_B_len = len(train_B_full)
    combined_B = pd.concat([train_B_full, test_B_full], ignore_index=True)
    processed_B, final_features_B_list = feature_engineer_B(combined_B)
    
    X_train_B = processed_B.iloc[:train_B_len].drop('Test_id', axis=1)
    X_test_B = processed_B.iloc[train_B_len:].drop('Test_id', axis=1)
    y_train_B = train_B_full['Label']
    test_B_ids = test_B_full['Test_id']

    
    print(f"B-전처리기(피처 목록) 저장 : {B_PREPROC_PATH}")
    joblib.dump(final_features_B_list, B_PREPROC_PATH)
    
    
    # (4) 모델 학습 (A, B)
    print("\n--- 4. 모델 학습 ---")
    
    print("Model A - Optuna 튜닝을 시작합니다.")
    X_train_A_opt, X_val_A_opt, y_train_A_opt, y_val_A_opt = train_test_split(
        X_train_A, y_train_A, test_size=0.2, random_state=42, stratify=y_train_A)
    
    # model A
    def objective_A(trial):
        params={
            'objective' : 'binary',
            'metric' : 'auc',
            'verbosity' : -1,
            'boosting_type' : 'gbdt',
            'class_weight' : 'balanced',
            'random_state' : 42,
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
            'max_depth' : trial.suggest_int('max_depth', 3, 10),
            'num_leaves' : trial.suggest_int('num_leaves', 20, 100),
            'subsample' : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        model = LGBMClassifier(**params)
        model.fit(X_train_A_opt, y_train_A_opt)
        proba_val = model.predict_proba(X_val_A_opt)[:, 1]
        brier_loss = brier_score_loss(y_val_A_opt, proba_val)
        return brier_loss
    
    study_A = optuna.create_study(direction='minimize')
    study_A.optimize(objective_A, n_trials=15)
    
    best_params_A = study_A.best_params
    best_params_A['class_weight'] = 'balanced'
    best_params_A['random_state']=42
    
    print(f"Model A 최적 brier 점수 : {study_A.best_value:4f}")
    print(f"Model A 최적 파라미터 : {best_params_A}")
    
    model_A = LGBMClassifier(**best_params_A)
    model_A.fit(X_train_A, y_train_A)
    print("Model A 최종 학습 완료.")
    
    print(f"A-모델 저장 : {A_MODEL_PATH}")
    joblib.dump(model_A, A_MODEL_PATH)
    
    
    # Model B
    print("\nModel B - Optuna 튜닝을 시작합니다.")
    X_train_B_opt, X_val_B_opt, y_train_B_opt, y_val_B_opt = train_test_split(
    X_train_B, y_train_B, test_size=0.2, random_state=42, stratify=y_train_B
    )
    
    def objective_B(trial):
        params={
            'objective' : 'binary',
            'metric' : 'auc',
            'verbosity' : -1,
            'boosting_type' : 'gbdt',
            'class_weight' : 'balanced',
            'random_state' : 42,
            'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
            'max_depth' : trial.suggest_int('max_depth', 3, 10),
            'num_leaves' : trial.suggest_int('num_leaves', 20, 100),
            'subsample' : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        model = LGBMClassifier(**params)
        model.fit(X_train_B_opt, y_train_B_opt)
        proba_val = model.predict_proba(X_val_B_opt)[:,1]
        brier_loss = brier_score_loss(y_val_B_opt, proba_val)
        return brier_loss
    study_B = optuna.create_study(direction='minimize')
    study_B.optimize(objective_B, n_trials=15)
    
    best_params_B = study_B.best_params
    best_params_B['class_weight'] = 'balanced'
    best_params_B['random_state'] = 42
    
    print(f"Model B 최적 Brier 점수 : {study_B.best_value:.4f}")
    print(f"Model B 최적 파라미터 : {best_params_B}")
    
    model_B = LGBMClassifier(**best_params_B)
    model_B.fit(X_train_B, y_train_B)
    print("Model B 최종 학습 완료")
    print(f"B-모델 저장 : {B_MODEL_PATH}")
    joblib.dump(model_B, B_MODEL_PATH)
    
#     # Model A
#     model_A = LGBMClassifier(random_state=42, class_weight='balanced')
#     model_A.fit(X_train_A, y_train_A)
#     print("Model A 학습 완료.")
    
#     print(f"A-모델 저장: {A_MODEL_PATH}")
#     joblib.dump(model_A, A_MODEL_PATH)
    
    
#     # Model B
#     model_B = LGBMClassifier(random_state=42, class_weight='balanced')
#     model_B.fit(X_train_B, y_train_B)
#     print("Model B 학습 완료.")

#     print(f"B-모델 저장: {B_MODEL_PATH}")
#     joblib.dump(model_B, B_MODEL_PATH)
    
    
    # (5) 예측
    print("\n--- 5. 예측 ---")
    preds_A = model_A.predict_proba(X_test_A)[:, 1]
    preds_B = model_B.predict_proba(X_test_B)[:, 1]

    # (6) 제출 파일 생성
    sub_A = pd.DataFrame({'Test_id': test_A_ids, 'Label': preds_A})
    sub_B = pd.DataFrame({'Test_id': test_B_ids, 'Label': preds_B})
    
    sub_A['Test_id'] = sub_A['Test_id'].astype(str)
    sub_B['Test_id'] = sub_B['Test_id'].astype(str)
    test_main['Test_id'] = test_main['Test_id'].astype(str)
    
    submission_preds = pd.concat([sub_A, sub_B], axis=0)
    
    # test_main의 ID 순서와 일치시키기 (DACON 규정)
    submission_final = pd.merge(test_main[['Test_id']], submission_preds, on='Test_id', how='left')
    
    submission_preds['Test_id'] = submission_preds['Test_id'].astype(str)
    sample_submission['Test_id'] = sample_submission['Test_id'].astype(str)
    
    if 'Label' in sample_submission.columns:
        sample_submission = sample_submission.drop('Label', axis=1)
    submission_final = pd.merge(sample_submission, submission_preds, on='Test_id', how='left')
    

    submission_final['Label'] = pd.to_numeric(submission_final['Label'], errors='coerce')
    submission_final['Label'] = submission_final['Label'].fillna(0.5)
    submission_final['Label'] = submission_final['Label'].clip(0, 1)
    submission_final['Label'] = submission_final['Label'].astype(float)
    submission_final.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"메타데이터 저장 : {META_PATH}")
    meta_data={
        "model_A_params" : model_A.get_params(),
        "model_B_params" : model_B.get_params(),
        "features_A" : final_features_A_list,
        "features_B" : final_features_B_list
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta_data, f, indent=4)
        
    print(f"\n--- 6. 완료 ---")
    print(f"제출 파일이 '{SUBMISSION_PATH}'에 성공적으로 저장되었습니다.")
    print("제출 파일 샘플:")
    print(submission_final.head())

# --- 스크립트 실행 ---
if __name__ == "__main__":
    main()

