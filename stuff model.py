import pandas as pd 
import numpy as np
import itertools
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
drop_cols = ["b_ss", "Unnamed: 0", "prev_pitch"]
x = model_data.drop(columns=drop_cols)
y = model_data["b_ss"]
test_size = 0.33
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rng, stratify=y)

# remove keys from training sets
keys = ['game_pk', 'pitcher', 'game_year', 'player_name', 'cluster', 'at_bat_number', 'pitch_number', 'pitch_type', 'b_right', 'p_right']
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

##### SKIP IF ALREADY RAN ######

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

param_sampler = ParameterSampler(param_grid, n_iter=300, random_state=rng)
tune_grid = pd.DataFrame(param_sampler)
tune_params = []

# ordinal loss score function (cohen's kappa)
def early_stopping_loss(y_true, y_pred):
    kappa = cohen_kappa_score(y_true, y_pred.astype(int), weights='quadratic')
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

####### RESUME HERE ######

# run final model
opt_params = pd.read_csv("opt_params.csv")
opt_params.set_index('Unnamed: 0', inplace=True)
opt_params = opt_params.T.reset_index()

model = LGBMOrdinal(n_jobs=2, importance_type="gain", random_state=rng,
                    n_estimators=opt_params['n_estimators'].astype(int).iloc[0],
                    learning_rate=opt_params['learning_rate'].iloc[0],
                    max_depth=opt_params['max_depth'].astype(int).iloc[0],
                    num_leaves=opt_params['num_leaves'].astype(int).iloc[0],
                    min_split_gain=opt_params['min_split_gain'].iloc[0],
                    min_child_weight=opt_params['min_child_weight'].iloc[0],
                    min_child_samples=opt_params['min_child_samples'].astype(int).iloc[0],
                    subsample=0.5,
                    subsample_freq=opt_params['subsample_freq'].astype(int).iloc[0],
                    colsample_bytree=0.5,
                    # class_weight={0: 4/3, 1: 2/3, 2: 4/3},
                    reg_alpha=opt_params['reg_alpha'].iloc[0],
                    reg_lambda=opt_params['reg_lambda'].iloc[0])
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

# create deception scores
model_val['deception_score'] = ((model_val['prob_0']) + (model_val['prob_1'] * 1.18) + (model_val['prob_2'] * 1.72)) # lower = better
model_val['deception_score_plus'] = 100 * (np.mean(model_val['deception_score']) / model_val['deception_score'])

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
plt.show()
plt.savefig('shap_summary_plot.png', bbox_inches='tight')
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
#opt_params.to_csv('opt_params.csv')`

del y_train, y_test, y_resampled, y, x_train, x_test, x_resampled, x_keys, x

####################################
# Get deception Score for All Pitchers #
####################################

model_data_test = pd.read_csv("stuff_model_test.csv")
pitch_mapping = {
    '4-Seam Fastball': 'FF',
    'Changeup': 'CH',
    'Curveball': 'CU',
    'Cutter': 'FC',
    'Sinker': 'SI',
    'Slider': 'SL',
    'Split-Finger': 'FS',
    'Sweeper': 'ST',
    'SV': 'SL',
    'KC': 'CU'
}
# Fix Pitch Types
model_data_test['prev_pitch'] = model_data_test['prev_pitch'].map(pitch_mapping)
model_data_test['pitch_type'] = model_data_test['pitch_type'].map(pitch_mapping).fillna(model_data_test['pitch_type'])
model_pred = model_data_test.drop(columns=['Unnamed: 0', 'b_ss', 'game_pk', 'pitcher', 'game_year', 'player_name', 'cluster', 'at_bat_number', 'pitch_number', 'pitch_type', 'prev_pitch', 'b_right', 'p_right'])

all_pred = model.predict(model_pred)
all_prob = model.predict_proba(model_pred)

all_deception = model_data_test.drop(columns=['Unnamed: 0'])
all_deception['pred'] = all_pred
all_deception['prob_0'] = all_prob[:, 0]
all_deception['prob_1'] = all_prob[:, 1]
all_deception['prob_2'] = all_prob[:, 2]
all_deception['deception_score'] = ((all_deception['prob_0']) + (all_deception['prob_1'] * 1.18) + (all_deception['prob_2'] * 1.72)) # lower = better

# Standardize the deception_score with a larger disparity
mean_deception_score = all_deception['deception_score'].mean()
std_deception_score = all_deception['deception_score'].std()
scaling_factor = 15  # Change the scaling factor to create a larger disparity

# Transform to the new scale where 100 is the average
all_deception['deception_score_plus'] = 100 + (all_deception['deception_score'] - mean_deception_score) * (scaling_factor / std_deception_score)
all_deception['deception_score_plus'] = 200 - all_deception['deception_score_plus']

all_deception = all_deception.dropna(subset=['cluster', 'p_right', 'b_right'])

pitcher_deception = all_deception.groupby(['pitcher', 'player_name', 'game_year', 'cluster', 'p_right', 'b_right', 'pitch_type']).agg(
    pitcher_size=('pitcher', 'size'),
    release_speed_mean=('release_speed', 'mean'),
    pfx_x_mean=('pfx_x', 'mean'),
    pfx_z_mean=('pfx_z', 'mean'),
    arm_angle_mean=('arm_angle', 'mean'),
    release_pos_z_mean=('release_pos_z', 'mean'),
    pf_velo_diff=('pf_velo_diff', 'mean'),
    pf_hbreak_diff=('pf_hbreak_diff', 'mean'),
    pf_vbreak_diff=('pf_vbreak_diff', 'mean'),
    pf_aa_diff=('pf_aa_diff', 'mean'),
    prob_0_mean=('prob_0', 'mean'),
    prob_1_mean=('prob_1', 'mean'),
    prob_2_mean=('prob_2', 'mean'),
    deception_score_plus_mean=('deception_score_plus', 'mean'),
    deception_score_plus_min=('deception_score_plus', 'min'),
    deception_score_plus_max=('deception_score_plus', 'max'),
).reset_index()


####################################
   # Load in Predicted Pitches #
####################################

new_pitch_proj = pd.read_csv("pitch_projections.csv")

####################################
# Finding Previously Thrown Pitches #
####################################

unique_pitch_types = all_deception['pitch_type'].unique()
unique_b_right = all_deception['b_right'].unique()

# Generate all possible combinations of pitch_type, prev_pitch, and b_right
all_combinations = list(itertools.product(unique_pitch_types, unique_pitch_types, unique_b_right))
combinations_df = pd.DataFrame(all_combinations, columns=['prev_pitch', 'pitch_type', 'b_right'])

# Get unique combinations of pitcher, p_right, and cluster
unique_pitchers_info = all_deception[['pitcher', 'p_right', 'cluster']].drop_duplicates()

combined_data = pd.concat([model_data, model_data_test])
last_thrown = combined_data.groupby(['pitcher', 'pitch_type', 'cluster'])['game_year'].max().reset_index()
last_thrown.columns = ['pitcher', 'pitch_type', 'cluster', 'last_season']
last_thrown = last_thrown[last_thrown['pitcher'].isin(unique_pitchers_info['pitcher'])]
prev_thrown = last_thrown[last_thrown['last_season'] != 2023]

####################################
  # Create Dataset for Matricies #
####################################

# Create a Cartesian product of unique pitcher info and all combinations
matrix_df = pd.DataFrame(itertools.product(unique_pitchers_info.itertuples(index=False, name=None), all_combinations),
                         columns=['pitcher_info', 'combination'])

# Expand the tuples into separate columns
matrix_df[['pitcher', 'p_right', 'cluster']] = pd.DataFrame(matrix_df['pitcher_info'].tolist(), index=matrix_df.index)
matrix_df[['prev_pitch', 'pitch_type', 'b_right']] = pd.DataFrame(matrix_df['combination'].tolist(), index=matrix_df.index)

# Drop the original tuple columns
matrix_df = matrix_df.drop(columns=['pitcher_info', 'combination']).drop_duplicates()

# Create the Matrix of all Pitches
matrix_df = matrix_df.dropna(subset=['cluster'])

# Get Currenty thrown pitches
pitches_deception = all_deception.groupby(['pitcher', 'pitch_type', 'prev_pitch', 'p_right', 'b_right', 'cluster']).agg(
    release_speed=('release_speed', 'mean'),
    pfx_x=('pfx_x', 'mean'),
    pfx_z=('pfx_z', 'mean'),
    arm_angle=('arm_angle', 'mean'),
    release_pos_z=('release_pos_z', 'mean'),
    pf_velo_diff=('pf_velo_diff', 'mean'),
    pf_hbreak_diff=('pf_hbreak_diff', 'mean'),
    pf_vbreak_diff=('pf_vbreak_diff', 'mean'),
    pf_aa_diff=('pf_aa_diff', 'mean'),
    velo_diff=('velo_diff', 'mean'),
    x_diff=('x_diff', 'mean'),
    y_diff=('y_diff', 'mean'),
    deception_plus=('deception_score_plus', 'mean')
).reset_index()
pitches_deception['predicted'] = 0

current_pitches = pd.merge(matrix_df, pitches_deception, how='left', 
                     left_on=['pitcher', 'pitch_type', 'prev_pitch', 'p_right', 'b_right', 'cluster'], 
                     right_on = ['pitcher', 'pitch_type', 'prev_pitch', 'p_right', 'b_right', 'cluster'])

# Find the pitches needed to predict
pitch_to_predict = current_pitches[current_pitches['deception_plus'].isna()]
pitch_to_predict = pitch_to_predict[['pitcher', 'pitch_type', 'prev_pitch', 'p_right', 'b_right', 'cluster']]

thrown_pitches = all_deception.groupby(['pitcher', 'pitch_type', 'p_right', 'b_right', 'cluster']).agg(
    release_speed=('release_speed', 'mean'),
    pfx_x=('pfx_x', 'mean'),
    pfx_z=('pfx_z', 'mean'),
    arm_angle=('arm_angle', 'mean'),
    release_pos_z=('release_pos_z', 'mean'),
    pf_velo_diff=('pf_velo_diff', 'mean'),
    pf_hbreak_diff=('pf_hbreak_diff', 'mean'),
    pf_vbreak_diff=('pf_vbreak_diff', 'mean'),
    pf_aa_diff=('pf_aa_diff', 'mean'),
).reset_index()

# Grab all pitches previously thrown
previously_thrown_pitches = combined_data.groupby(['pitcher', 'pitch_type', 'p_right', 'b_right', 'cluster']).agg(
    release_speed=('release_speed', 'mean'),
    pfx_x=('pfx_x', 'mean'),
    pfx_z=('pfx_z', 'mean'),
    arm_angle=('arm_angle', 'mean'),
    release_pos_z=('release_pos_z', 'mean'),
    pf_velo_diff=('pf_velo_diff', 'mean'),
    pf_hbreak_diff=('pf_hbreak_diff', 'mean'),
    pf_vbreak_diff=('pf_vbreak_diff', 'mean'),
    pf_aa_diff=('pf_aa_diff', 'mean'),
).reset_index()
prev_thrown_tuples = set(prev_thrown[['pitcher', 'pitch_type', 'cluster']].itertuples(index=False, name=None))
previously_thrown_pitches = previously_thrown_pitches[
    previously_thrown_pitches.set_index(['pitcher', 'pitch_type', 'cluster']).index.isin(prev_thrown_tuples)
]

# Combine all pitches together
all_pitches = pd.concat([thrown_pitches, new_pitch_proj, previously_thrown_pitches], ignore_index=True)

# Adding pitches where there is only 1 obs (no same or opposite handedness)
single_obs_pitches = all_pitches.groupby(['pitcher', 'pitch_type', 'cluster']).filter(lambda x: len(x) == 1)

# Duplicate and modify
duplicated_pitches = single_obs_pitches.copy()
duplicated_pitches['pfx_x'] = -duplicated_pitches['pfx_x']
duplicated_pitches['b_right'] = np.where(duplicated_pitches['b_right'] == 1, 0, 1)

# Grab Primary Fastball horizontal movement
pf_fb_horz = pitcher_deception[pitcher_deception['pf_velo_diff'].isna()]
pf_fb_horz = pf_fb_horz[['pitcher', 'cluster', 'p_right', 'b_right', 'pfx_x_mean']]

# Merge duplicated_pitches with pf_fb_horz
duplicated_pitches = pd.merge(duplicated_pitches, pf_fb_horz, how='left',
                              left_on=['pitcher', 'p_right', 'b_right', 'cluster'],
                              right_on=['pitcher', 'p_right', 'b_right', 'cluster'])

# Calculate the pf_hbreak_diff column
duplicated_pitches['pf_hbreak_diff'] = duplicated_pitches['pfx_x'] - duplicated_pitches['pfx_x_mean']

# Drop the 'pfx_x_mean' column as it is no longer needed
duplicated_pitches = duplicated_pitches.drop(columns=['pfx_x_mean'])

# Validation test of new pitches
val_dupes = pd.concat([single_obs_pitches, duplicated_pitches], ignore_index=True)

# Concatenate the modified duplicated pitches
all_pitches = pd.concat([all_pitches, duplicated_pitches], ignore_index=True)

predicted_deception = pd.merge(
    all_pitches, all_pitches, 
    how='left', 
    left_on=['pitcher', 'p_right', 'b_right', 'cluster'],
    right_on=['pitcher', 'p_right', 'b_right', 'cluster']
)
predicted_deception['velo_diff'] = predicted_deception['release_speed_y'] - predicted_deception['release_speed_x']
predicted_deception['x_diff'] = predicted_deception['pfx_x_y'] - predicted_deception['pfx_x_x']
predicted_deception['y_diff'] = predicted_deception['pfx_z_y'] - predicted_deception['pfx_z_x']
selected_columns = [
    'pitcher',
    'pitch_type_x',
    'pitch_type_y',
    'p_right',
    'b_right',
    'cluster',
    'release_speed_x',
    'pfx_x_x',
    'pfx_z_x',
    'arm_angle_x',
    'release_pos_z_x',
    'pf_velo_diff_x',
    'pf_hbreak_diff_x',
    'pf_vbreak_diff_x',
    'pf_aa_diff_x',
    'velo_diff',
    'x_diff',
    'y_diff'
]

predicted_deception = predicted_deception[selected_columns].rename(
    columns={
        'pitch_type_x': 'pitch_type',
        'pitch_type_y': 'prev_pitch',
        'release_speed_x': 'release_speed',
        'pfx_x_x': 'pfx_x',
        'pfx_z_x': 'pfx_z',
        'arm_angle_x': 'arm_angle',
        'release_pos_z_x': 'release_pos_z',
        'pf_velo_diff_x': 'pf_velo_diff',
        'pf_hbreak_diff_x': 'pf_hbreak_diff',
        'pf_vbreak_diff_x': 'pf_vbreak_diff',
        'pf_aa_diff_x': 'pf_aa_diff',
        'velo_diff': 'velo_diff',
        'x_diff': 'x_diff',
        'y_diff': 'y_diff'
    }
)

deception_pred = predicted_deception.drop(columns=['pitcher', 'cluster', 'pitch_type', 'prev_pitch', 'b_right', 'p_right'])

pred = model.predict(deception_pred)
prob = model.predict_proba(deception_pred)

predicted_deception['pred'] = pred
predicted_deception['prob_0'] = prob[:, 0]
predicted_deception['prob_1'] = prob[:, 1]
predicted_deception['prob_2'] = prob[:, 2]
predicted_deception['deception_score'] = ((predicted_deception['prob_0']) + (predicted_deception['prob_1'] * 1.18) + (predicted_deception['prob_2'] * 1.72)) # lower = better

# Standardize the deception_score with a larger disparity
mean_deception_score = predicted_deception['deception_score'].mean()
std_deception_score = predicted_deception['deception_score'].std()
scaling_factor = 15  # Change the scaling factor to create a larger disparity

# Transform to the new scale where 100 is the average
predicted_deception['deception_plus'] = 100 + (predicted_deception['deception_score'] - mean_deception_score) * (scaling_factor / std_deception_score)
predicted_deception['deception_plus'] = 200 - predicted_deception['deception_plus']
predicted_deception = predicted_deception.drop(columns=['pred', 'prob_0', 'prob_1', 'prob_2', 'deception_score'])
# Create Dummy for previously thrown / predicted
predicted_deception_tuples = set(predicted_deception[['pitcher', 'pitch_type', 'cluster']].itertuples(index=False, name=None))
prev_thrown_tuples = set(previously_thrown_pitches[['pitcher', 'pitch_type', 'cluster']].itertuples(index=False, name=None))
predicted_deception['predicted'] = np.where(
    predicted_deception.set_index(['pitcher', 'pitch_type', 'cluster']).index.isin(prev_thrown_tuples),
    2,
    1
)

pitch_to_predict = pd.merge(pitch_to_predict, 
                            predicted_deception, 
                            how='left',
                            left_on=['pitcher', 'pitch_type', 'prev_pitch', 'p_right', 'b_right', 'cluster'], 
                            right_on = ['pitcher', 'pitch_type', 'prev_pitch', 'p_right', 'b_right', 'cluster'])

all_pitches_deception = pd.concat([current_pitches, predicted_deception], ignore_index=True)

def filter_group(group):
    if len(group) > 1:
        if (group['predicted'] == 0).any():
            return group[group['predicted'] == 0]
        elif (group['predicted'] == 2).any():
            return group[group['predicted'] == 2]
        else:
            return group[group['predicted'] == 1]
    else:
        return None

# Initialize lists to store the results
filtered_groups = []
deception_plus_diff = []

# Group by the specified columns
grouped = all_pitches_deception.groupby(['pitcher', 'pitch_type', 'prev_pitch', 'p_right', 'b_right', 'cluster'])

for name, group in grouped:
    # Apply filtering logic
    filtered_group = filter_group(group)
    filtered_groups.append(filtered_group)

# Concatenate the filtered groups back into a single DataFrame
all_pitches_filtered = pd.concat(filtered_groups).drop_duplicates().reset_index(drop=True)

matrix_df = pd.merge(matrix_df, all_pitches_filtered, how='left', 
                     left_on=['pitcher', 'pitch_type', 'prev_pitch', 'p_right', 'b_right', 'cluster'], 
                     right_on = ['pitcher', 'pitch_type', 'prev_pitch', 'p_right', 'b_right', 'cluster']).drop_duplicates()
matrix_df['pitch_percentile'] = matrix_df.groupby('pitch_type')['deception_plus'].rank(pct=True) * 100

matrix_df.to_csv("deception_matrix_df.csv", index=False)

temp = combined_data[combined_data['pitcher'] == 518886]

####################################
       # Grab NA Pitches #
####################################

na_pitches = matrix_df[matrix_df.isna()['release_speed']]
na_pitch_counts = na_pitches.groupby(['pitcher', 'cluster', 'pitch_type']).size().reset_index(name='count')
na_to_predict = na_pitch_counts[na_pitch_counts['count'] == 16]
na_to_predict.to_csv("na_to_predict.csv")

####################################
   # Create Validation Data Set #
####################################

val_pred_data = pd.read_csv("val_set.csv")
val_pred_data['pitch_type'] = val_pred_data['pitch_type'].map(pitch_mapping)
val_pred_data['predicted'] = 1

# Grab other pitches each throws
val_other_pitches = all_pitches[all_pitches['pitcher'].isin(val_pred_data['pitcher'])]
val_other_pitches['predicted'] = 0
val_combined = pd.concat([val_other_pitches, val_pred_data], ignore_index=True)
grouped = val_combined.groupby(['pitcher', 'pitch_type', 'b_right', 'p_right', 'cluster'])
val_other_pitches = grouped.apply(lambda x: x[x['predicted'] == 1] if len(x) > 1 else x).reset_index(drop=True)

val_pred_data = pd.merge(
    val_pred_data, val_other_pitches, 
    how='left', 
    left_on=['pitcher', 'p_right', 'b_right', 'cluster'],
    right_on=['pitcher', 'p_right', 'b_right', 'cluster']
)
val_pred_data['velo_diff'] = val_pred_data['release_speed_y'] - val_pred_data['release_speed_x']
val_pred_data['x_diff'] = val_pred_data['pfx_x_y'] - val_pred_data['pfx_x_x']
val_pred_data['y_diff'] = val_pred_data['pfx_z_y'] - val_pred_data['pfx_z_x']
selected_columns = [
    'pitcher',
    'pitch_type_x',
    'pitch_type_y',
    'p_right',
    'b_right',
    'cluster',
    'release_speed_x',
    'pfx_x_x',
    'pfx_z_x',
    'arm_angle_x',
    'release_pos_z_x',
    'pf_velo_diff_x',
    'pf_hbreak_diff_x',
    'pf_vbreak_diff_x',
    'pf_aa_diff_x',
    'velo_diff',
    'x_diff',
    'y_diff',
    'predicted_x',
    'predicted_y'
]

val_pred_data = val_pred_data[selected_columns].rename(
    columns={
        'pitch_type_x': 'pitch_type',
        'pitch_type_y': 'prev_pitch',
        'release_speed_x': 'release_speed',
        'pfx_x_x': 'pfx_x',
        'pfx_z_x': 'pfx_z',
        'arm_angle_x': 'arm_angle',
        'release_pos_z_x': 'release_pos_z',
        'pf_velo_diff_x': 'pf_velo_diff',
        'pf_hbreak_diff_x': 'pf_hbreak_diff',
        'pf_vbreak_diff_x': 'pf_vbreak_diff',
        'pf_aa_diff_x': 'pf_aa_diff',
        'velo_diff': 'velo_diff',
        'x_diff': 'x_diff',
        'y_diff': 'y_diff',
        'predicted_x': 'predicted_current',
        'predicted_y': 'predicted_prev'
    }
).reset_index()

val_pred = val_pred_data.drop(['index', 'pitcher', 'cluster', 'pitch_type', 'prev_pitch', 'b_right', 'p_right', 'predicted_current', 'predicted_prev'], axis=1, errors='ignore')

pred_val = model.predict(val_pred)
prob_val = model.predict_proba(val_pred)

validation = val_pred_data
validation['pred'] = pred_val
validation['prob_0'] = prob_val[:, 0]
validation['prob_1'] = prob_val[:, 1]
validation['prob_2'] = prob_val[:, 2]
validation['deception_score'] = ((validation['prob_0']) + (validation['prob_1'] * 1.18) + (validation['prob_2'] * 1.72)) # lower = better

# Standardize the deception_score with a larger disparity
mean_deception_score = validation['deception_score'].mean()
std_deception_score = validation['deception_score'].std()
scaling_factor = 15  # Change the scaling factor to create a larger disparity

# Transform to the new scale where 100 is the average
validation['deception_plus'] = 100 + (validation['deception_score'] - mean_deception_score) * (scaling_factor / std_deception_score)
validation['deception_plus'] = 200 - validation['deception_plus']
validation = validation.drop(columns=['index', 'pred', 'prob_0', 'prob_1', 'prob_2', 'deception_score'])

validation_counts = model_data_test.groupby(['pitcher', 'cluster', 'pitch_type', 'prev_pitch', 'p_right', 'b_right']).size().reset_index(name='pitches')
validation = pd.merge(validation, validation_counts, how='left', 
                      left_on=['pitcher', 'p_right', 'b_right', 'cluster', 'pitch_type', 'prev_pitch'],
                      right_on=['pitcher', 'p_right', 'b_right', 'cluster', 'pitch_type', 'prev_pitch'])

validation_merge = validation[['pitcher', 'p_right', 'b_right', 'cluster', 'pitch_type', 'prev_pitch', 'deception_plus', 'pitches']]
validation_analysis = pd.merge(matrix_df, validation_merge, how='right', 
                      left_on=['pitcher', 'p_right', 'b_right', 'cluster', 'pitch_type', 'prev_pitch'],
                      right_on=['pitcher', 'p_right', 'b_right', 'cluster', 'pitch_type', 'prev_pitch'])
validation_analysis = validation_analysis[validation_analysis['pitches'] >= 20]
validation_analysis['e'] = abs(validation_analysis['deception_plus_y'] - validation_analysis['deception_plus_x']) <= 5

# Error Condition
pct_accurate = (validation_analysis['e'].sum() / len(validation_analysis)) * 100

# Accuracy Rate: 68%

####################################
  # Marginal Effects of # Pitches #
####################################

Pitch_Deception = all_deception.groupby(['pitcher', 'cluster', 'p_right', 'b_right', 'pitch_type', 'prev_pitch']).agg(
    pitches_thrown=('pitcher', 'count'),
    deception_plus_mean=('deception_score_plus', 'mean'),
    deception_plus_min=('deception_score_plus', 'min'),
    deception_plus_max=('deception_score_plus', 'max'),
).reset_index()

def weighted_mean(data, weights):
    return (data * weights).sum() / weights.sum()

def arsenal_calculation(group):
    weighted_deception = weighted_mean(group['deception_plus_mean'], group['pitches_thrown'])
    total_pitches = group['pitches_thrown'].sum()
    unique_pitches = group['pitch_type'].nunique()
    return pd.Series({
        'weighted_deception_plus': weighted_deception,
        'total_observations': total_pitches,
        'unique_pitches': unique_pitches
    })

Arsenal_Deception = Pitch_Deception.groupby(['pitcher', 'cluster', 'p_right', 'b_right']).apply(arsenal_calculation).reset_index()
Arsenal_Deception = Arsenal_Deception[Arsenal_Deception['unique_pitches'] > 1]

import statsmodels.api as sm

###### Linear Regression ########
Arsenal_Deception['unique_pitches'] = Arsenal_Deception['unique_pitches'].astype('category')
X = pd.get_dummies(Arsenal_Deception[['unique_pitches']], drop_first=True)
X = X.astype(int) # Convert boolean columns to integer
y = Arsenal_Deception['weighted_deception_plus']
X = sm.add_constant(X)

marginal_effects = sm.OLS(y, X).fit()

# Print the summary
print(marginal_effects.summary())

summary_df = pd.DataFrame({
    'Coefficient': marginal_effects.params,
    'Standard Error': marginal_effects.bse,
    't-Value': marginal_effects.tvalues,
    'P-Value': marginal_effects.pvalues
})

# Add a constant column if needed (for the intercept)
if 'const' in summary_df.index:
    summary_df.loc['const', 'Coefficient'] = marginal_effects.params['const']
    summary_df.loc['const', 'Standard Error'] = marginal_effects.bse['const']
    summary_df.loc['const', 't-Value'] = marginal_effects.tvalues['const']
    summary_df.loc['const', 'P-Value'] = marginal_effects.pvalues['const']
    
summary_df.to_csv('marginal_effects.csv', index=True)
