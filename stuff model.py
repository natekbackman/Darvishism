import pandas as pd 
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
from ordinalgbt.lgb import LGBMOrdinal
from ordinalgbt.loss import lgb_ordinal_loss
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# load data
model_data = pd.read_csv("stuff_model.csv")
seed = datetime.now().hour * 100 + datetime.now().minute
rng = np.random.RandomState(seed=seed)

# df dimensions and dtypes
print(model_data.shape)
print(model_data.dtypes)

# split data into response and predictors and train/test
drop_cols = ["b_ss", "Unnamed: 0"]
x = model_data.drop(columns=drop_cols)
y = model_data["b_ss"]
test_size = 0.33
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rng, stratify=y)

# remove keys from training sets
keys = ['game_pk', 'pitcher', 'at_bat_number', 'pitch_number']
x_keys = x_test[keys]
x_train = x_train.drop(columns=keys)
x_test = x_test.drop(columns=keys)

# dependent variable class distribution
y_train.value_counts()
y_test.value_counts()

# basic undersampling
rus = RandomUnderSampler(random_state=rng)
x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
y_resampled.value_counts()

# cross validation attempt
from sklearn.model_selection import KFold

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=rng)

# tune params
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

# select params from LGBMOrdinal
param_grid = {
    'n_estimators': randint(100, 1000),      # Number of boosting rounds
    'learning_rate': uniform(0.005, 0.2),    # Step size shrinkage
    'max_depth': randint(3, 15),             # Maximum depth of trees
    'num_leaves': randint(10, 100),          # Maximum number of leaves per tree
    'min_split_gain': uniform(0, 0.5),       # Minimum loss reduction for further partition
    'min_child_weight': uniform(0.1, 10),    # Minimum sum of instance weight (hessian) needed in a child
    'min_child_samples': randint(5, 50),     # Minimum number of data needed in a child
    # 'subsample': uniform(0.5, 0.5),          # Subsample ratio of the training instance
    'subsample_freq': randint(1, 10),        # Frequency of subsample, <=0 means no subsampling
    # 'colsample_bytree': uniform(0.5, 0.5),   # Subsample ratio of columns when constructing each tree
    'reg_alpha': uniform(0, 0.1),            # L1 regularization term on weights
    'reg_lambda': uniform(0, 0.1)            # L2 regularization term on weights
}

param_sampler = ParameterSampler(param_grid, n_iter=100, random_state=rng)
tune_grid = pd.DataFrame(param_sampler)
tune_params = []

# ordinal loss score function (cohen's kappa)
def early_stopping_loss(y_true, y_pred):
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return 'kappa', kappa, True

scalar_min = tune_grid['learning_rate'].min()
scalar_max = tune_grid['learning_rate'].max()

for params in param_sampler:
    fold_kappa = []
    fold_f1 = []

    scalar = (params['learning_rate'] - scalar_min) / (scalar_max - scalar_min) * (0.2 - 0.1) + 0.1
    early_stopping = lgb.early_stopping(stopping_rounds=int(np.floor(scalar * params['n_estimators'])))
    best_iterations = []

    for train_index, val_index in kf.split(x_resampled):
        X_train, X_val = x_resampled.iloc[train_index], x_resampled.iloc[val_index]
        Y_train, Y_val = y_resampled.iloc[train_index], y_resampled.iloc[val_index]
        # train_data = lgb.Dataset(X_train, label=Y_train)
        # val_data = lgb.Dataset(X_val, label=Y_val, reference=train_data)

        # Train the model
        cv_model = LGBMOrdinal(n_jobs=2, random_state=rng, 
                               subsample=0.5,
                               colsample_bytree=0.5,
                               n_estimators=params['n_estimators'],
                               learning_rate=params['learning_rate'],
                               max_depth=params['max_depth'],
                               num_leaves=params['num_leaves'],
                               min_split_gain=params['min_split_gain'],
                               min_child_weight=params['min_child_weight'],
                               min_child_samples=params['min_child_samples'],
                               subsample_freq=params['subsample_freq'],
                               reg_alpha=params['reg_alpha'],
                               reg_lambda=params['reg_lambda'])
        cv_model.fit(X_train, Y_train, callbacks=[early_stopping], 
                     eval_metric=lambda y_true, y_pred: early_stopping_loss(y_true, y_pred),
                     # eval_metric='logloss',
                     eval_set=[(X_val, Y_val)])
        best_iteration = cv_model.best_iteration_

        # Predict and evaluate on validation set
        val_pred = cv_model.predict(X_val)
        kappa = cohen_kappa_score(Y_val, val_pred, weights='quadratic')
        fold_kappa.append(kappa)
        micro_f1 = f1_score(Y_val, val_pred, average='micro')
        fold_f1.append(micro_f1)
        best_iterations.append(best_iteration)

    # Compute mean f1, kappa, and optimal rounds from early stopping across folds
    mean_kappa = np.mean(fold_kappa)
    mean_f1 = np.mean(fold_f1)
    mean_opt_rounds = np.mean(best_iterations)

    row = {'n_estimators': params['n_estimators'], 'learning_rate': params['learning_rate'],
           'max_depth': params['max_depth'], 'num_leaves': params['num_leaves'],
           'min_split_gain': params['min_split_gain'], 'min_child_weight': params['min_child_weight'],
           'min_child_samples': params['min_child_samples'], 'subsample_freq': params['subsample_freq'],
           'reg_alpha': params['reg_alpha'], 'reg_lambda': params['reg_lambda'], 
           'quad_kappa': mean_kappa, 'micro_f1': mean_f1, 'mean_opt_rounds': mean_opt_rounds}
    tune_params.append(row)
    # end of loop

opt_params = pd.DataFrame(tune_params).sort_values(by='quad_kappa', ascending=False).iloc[0]

# run final model
model = LGBMOrdinal(n_jobs=2, importance_type="gain", random_state=rng,
                    n_estimators=opt_params['n_estimators'].astype(int),
                    learning_rate=opt_params['learning_rate'],
                    max_depth=opt_params['max_depth'].astype(int),
                    num_leaves=opt_params['num_leaves'].astype(int),
                    min_split_gain=opt_params['min_split_gain'],
                    min_child_weight=opt_params['min_child_weight'],
                    min_child_samples=opt_params['min_child_samples'].astype(int),
                    subsample=0.5,
                    subsample_freq=opt_params['subsample_freq'].astype(int),
                    colsample_bytree=0.5,
                    # class_weight={0: 4/3, 1: 2/3, 2: 4/3},
                    reg_alpha=opt_params['reg_alpha'],
                    reg_lambda=opt_params['reg_lambda'])
model.fit(x_resampled, y_resampled)

# test pred data
test_pred = model.predict(x_test)
test_prob = model.predict_proba(x_test)
model_val = pd.DataFrame({
    'actual': y_test,
    'pred': test_pred,
    'prob_0': test_prob[:,0],
    'prob_1': test_prob[:,1],
    'prob_2': test_prob[:,2]
})
model_val = pd.concat([model_val, x_keys], axis=1)

# prediction class distribution
model_val['pred'].value_counts()

# final model kappa and f1 score
cohen_kappa_score(y_test, test_pred, weights='quadratic') # use quadrtic for heavier punishment on larger discrepancies 
f1_score(y_test, test_pred, average='micro')

# ordinal loss score for class 0 (ordinal logistic negative log likelihood for whiff)
from ordinalgbt.loss import ordinal_logistic_nll

ordinal_logistic_nll(y_true=y_test, y_preds=test_prob.flatten(), theta=[0.5, 1.5]) # how can theta boundaries be tuned to represent ordinal nature of whiff, weak contact, barrel?

# create stuff scores
model_val['stuff_score'] = ((model_val['prob_0']) + (model_val['prob_1'] * 1.18) + (model_val['prob_2'] * 1.72)) # lower = better
model_val['stuff_score_plus'] = 100 * (np.mean(model_val['stuff_score']) / model_val['stuff_score'])

# variable importance
var_imp = pd.DataFrame({
    'variable': x_train.columns,
    'importance': model.feature_importances_
})

# shap values
import shap
from shap import TreeExplainer

explainer = TreeExplainer(model, model_output='raw') # argument when model_output="log_loss": data=shap_data
shap_values = explainer.shap_values(X=x_test, check_additivity=True) # argument when model_output="log_loss": y=shap_data.index

# create shap viz (needs tinkering)
shap.summary_plot(shap_values, x_test)
plt.savefig('shap_summary_plot.png')
shap.plots.bar(shap_values)
plt.savefig('shap_bar_plot.png')

# save shap values to csv
columns = x_test.columns.tolist()
test_shap = pd.DataFrame(shap_values, columns=columns)

# cbind model_val and shap values
model_val.reset_index(drop=True, inplace=True)
model_val = pd.concat([model_val, test_shap], axis=1)

# write validation set to csv
model_val.to_csv('test_val_w_shap.csv', index=False)

###########
# scratch #
###########

# NOTE: consider undersampling for whiff class (0)
# x.columns
opt_params.to_csv('opt_params.csv')