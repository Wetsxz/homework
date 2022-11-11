import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 数据读取与拼接
train_data = pd.read_csv(r'C:\Users\15643\Desktop\Data\newtrain.csv')
test_data = pd.read_csv(r'C:\Users\15643\Desktop\Data\dataA.csv')
data = pd.concat([train_data, test_data]).reset_index(drop=True)
data['f3'] = data['f3'].map({'low': 0, 'mid': 1, 'high': 2})

# 暴力Feature 位置
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for i in range(len(loc_f)):
    for j in range(i + 1, len(loc_f)):
        data[f'{loc_f[i]}+{loc_f[j]}'] = data[loc_f[i]] + data[loc_f[j]]
        data[f'{loc_f[i]}-{loc_f[j]}'] = data[loc_f[i]] - data[loc_f[j]]
        data[f'{loc_f[i]}*{loc_f[j]}'] = data[loc_f[i]] * data[loc_f[j]]
        data[f'{loc_f[i]}/{loc_f[j]}'] = data[loc_f[i]] / data[loc_f[j]]

# 暴力Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
for i in range(len(com_f)):
    for j in range(i + 1, len(com_f)):
        data[f'{com_f[i]}+{com_f[j]}'] = data[com_f[i]] + data[com_f[j]]
        data[f'{com_f[i]}-{com_f[j]}'] = data[com_f[i]] - data[com_f[j]]
        data[f'{com_f[i]}*{com_f[j]}'] = data[com_f[i]] * data[com_f[j]]
        data[f'{com_f[i]}/{com_f[j]}'] = data[com_f[i]] / data[com_f[j]]

#进行数据清洗
cat_columns = ['f3']
data = pd.concat([train_data, test_data])

for col in cat_columns:
    lb = LabelEncoder()
    lb.fit(data[col])
    train_data[col] = lb.transform(train_data[col])
    test_data[col] = lb.transform(test_data[col])

num_columns = [ col for col in train_data.columns if col not in ['id', 'label', 'f3']]
feature_columns = num_columns + cat_columns
target = 'label'

train = train_data[feature_columns]
label = train_data[target]
test = test_data[feature_columns]

def model_train(model, model_name, kfold=5):
    oof_preds = np.zeros((train.shape[0]))
    test_preds = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold)
    print(f"Model = {model_name}")
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]
        model.fit(x_train,y_train)
        y_pred = model.predict_proba(x_test)[:,1]
        oof_preds[test_index] = y_pred.ravel()
        auc = roc_auc_score(y_test,y_pred)
        print("- KFold = %d, auc = %.4f" % (k, auc))
        test_fold_preds = model.predict_proba(test)[:, 1]
        test_preds += test_fold_preds.ravel()
    print("Model = %s, AUC = %.4f" % (model_name, roc_auc_score(label, oof_preds)))
    return test_preds / kfold

gbc = GradientBoostingClassifier()
gbc_test_preds = model_train(gbc,"GradientBoostingClassifier",12)

train = train[:50000]
label = label[:50000]
# 训练测试分离
train_data = data[~data['label'].isna()]
test_data = data[data['label'].isna()]

features = [i for i in train_data.columns if i not in ['label',  'id']]
y = train_data['label']
KF = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)
feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'n_jobs': 30,
    'learning_rate': 0.05,
    'num_leaves': 2 ** 6,
    'max_depth': 8,
    'tree_learner': 'serial',
    'colsample_bytree': 0.8,
    'subsample_freq': 1,
    'subsample': 0.8,
    'num_boost_round': 5000,
    'max_bin': 255,
    'verbose': -1,
    'seed': 2021,
    'bagging_seed': 2021,
    'feature_fraction_seed': 2021,
    'early_stopping_rounds': 100,
}

oof_lgb = np.zeros(len(train_data))
predictions_lgb = np.zeros((len(test_data)))

# 模型训练
for fold_, (trn_idx, val_idx) in enumerate(KF.split(train_data.values, y.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train_data.iloc[trn_idx][features], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(train_data.iloc[val_idx][features], label=y.iloc[val_idx])
    num_round = 3000
    clf = lgb.train_data(
        params,
        trn_data,
        num_round,
        valid_sets=[trn_data, val_data],
        verbose_eval=100,
        early_stopping_rounds=50,
    )
    oof_lgb[val_idx] = clf.predict(train_data.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions_lgb[:] += clf.predict(test_data[features], num_iteration=clf.best_iteration) / 5
    feat_imp_df['imp'] += clf.feature_importance() / 5

print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
