#################################################################################################
#
# Implements the different metrics to evaluate the models and the generated data
#
#################################################################################################


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from seaborn import heatmap

from dataloader import Dataloader


REAL_COLOR = "blue"
FAKE_COLOR = "red"

# utility function
def one_hot_dataset(df : pd.DataFrame,
                    cat_features : list[str],
                    encoder : OneHotEncoder = None,
                    encode_target : bool = True,
                    target_name : str = None) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    From the dataset given as input, one hot encodes the categorical features and returns the new version of the dataset

    Parameters:
        - df: pandas dataframe, the dataset to one hot encode
        - cat_features: list of strings, the names of the categorical features in the dataset
        - encoder: OneHotEncoder, an already fitted encoder to use to one hot encode the dataset
        - encode_target: bool, whether to one hot encode the target variable, default is True
        - target_name: str, name of the target variable, default is None (only used if encode_target is False)

    Returns the one hot encoded dataset
    """
    cat_feats = [f for f in cat_features]
    if not encode_target:
        target = df[target_name]
        df = df.drop(columns=target_name, axis=1)
        if target_name in cat_feats:
            cat_feats.remove(target_name)
    cat_feat = df[cat_feats]
    df = df.drop(columns=cat_feats)
    df = df.reset_index(drop=True)

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore')    # if unknown category, map it to all 0s
        encoder.fit(cat_feat)
    
    cat_feat = encoder.transform(cat_feat).toarray()
    cat_feat = pd.DataFrame(cat_feat, columns=encoder.get_feature_names_out()).reset_index(drop=True)
    encoded = pd.concat([df, cat_feat], axis=1).astype(float)

    if not encode_target:
        encoded[target_name] = target

    columns = encoded.columns
    col_to_rename = {}
    bad_chars = ["[", "]", "<", " ", ", "]
    for col in columns:
        if any(char in col for char in bad_chars):
            col_to_rename[col] = ''.join(c for c in col if c not in bad_chars)
    if len(col_to_rename) > 0:
        encoded = encoded.rename(columns=col_to_rename)
    
    return encoded, encoder


#############################################################





def plot_distributions(real: Dataloader, fake: pd.DataFrame, show: bool = False, path: str = None):
    """
    Plots the distribution comparisons between the real and the fake data

    Inputs:
    - real: Dataloader, the real data
    - fake: pd.DataFrame, the fake data
    - show: bool, whether to show the plot or not, default=False
    - path: string, path to save the plot. If None, does not save the plot, default=None
    """
    cat_cols = real.cat_features.copy()
    num_cols = real.num_features
    dname = real.dname
    target = real.target
    task = real.task
    real = real.data

    n_fig_per_row = 4
    n_rows = len(real.columns) // n_fig_per_row + 1 if len(real.columns) % n_fig_per_row != 0 else len(real.columns) // n_fig_per_row

    fig, axs = plt.subplots(n_rows, n_fig_per_row, figsize=(n_rows * 6, n_fig_per_row * 6))
    # fig.suptitle(f"Distributions for dataset {dname}", fontsize=20)
    for i, col_name in enumerate(real.columns):
        row, col = i // n_fig_per_row, i % n_fig_per_row

        if (col_name in cat_cols) or (target == col_name and task == "classification"):
            real_counts = real[col_name].value_counts(normalize=True)
            fake_counts = fake[col_name].value_counts(normalize=True)

            # create value in fake if not present
            for val in real_counts.index.values:
                if val not in fake_counts.index.values:
                    fake_counts[val] = 0
            
            real_counts.plot(kind="bar", color=REAL_COLOR, alpha=0.5, label="real", ax=axs[row, col])
            fake_counts.plot(kind="bar", color=FAKE_COLOR, alpha=0.5, label="fake", ax=axs[row, col])
            axs[row, col].set_title(f"{col_name}")
            axs[row, col].legend(fontsize=15)
            axs[row, col].set_xlabel("")
            axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=90, ha='right')
            axs[row, col].tick_params(axis='y', labelsize=15)
        else:
            real_counts = real[col_name]
            fake_counts = fake[col_name]

            real_counts.plot(kind="hist", color=REAL_COLOR, alpha=0.5, label="real", density=True, bins=25, ax=axs[row, col])
            fake_counts.plot(kind="hist", color=FAKE_COLOR, alpha=0.5, label="fake", density=True, bins=25, ax=axs[row, col])
            axs[row, col].set_title(f"{col_name}")
            axs[row, col].legend(fontsize=15)
            axs[row, col].set_xlabel("")
            axs[row, col].tick_params(axis='y', labelsize=15)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    if show:
        plt.show()


def plot_pca():
    ...


def plot_correlation_matrix(real: Dataloader, fake: pd.DataFrame, show: bool = False, path: str = None):
    """
    Plots the correlation matrices for the real and the fake data

    Inputs:
    - real: Dataloader, the real data
    - fake: pd.DataFrame, the fake data
    - show: bool, whether to show the plot or not, default=False
    - path: string, path to save the plot. If None, does not save the plot, default=None
    """
    dname = real.dname
    real = real.data

    real_corr = real.corr(method="pearson", numeric_only=True)
    fake_corr = fake.corr(method="pearson", numeric_only=True)
    diff_corr = np.abs(real_corr - fake_corr)
    for i in range(len(diff_corr)):
        diff_corr.iloc[i, i] = 1.0
    real_mask = np.triu(real_corr)
    fake_mask = np.triu(fake_corr)
    diff_mask = np.triu(diff_corr)
    
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    ax = heatmap(real_corr, cmap="coolwarm", annot=True, vmax=1, vmin=-1, cbar=False, mask=real_mask)
    plt.title(f"{dname} (real)")
    plt.yticks(ticks=ax.get_yticks()[1:], labels=real_corr.columns[1:], rotation=0)
    plt.xticks(ticks=ax.get_xticks()[:-1], labels=real_corr.columns[:-1], rotation=45)

    plt.subplot(1, 3, 2)
    ax = heatmap(fake_corr, cmap="coolwarm", annot=True, vmax=1, vmin=-1, cbar=False, mask=fake_mask)
    plt.title(f"{dname} (fake)")
    plt.yticks([], labels=[])
    plt.xticks(ticks=ax.get_xticks()[:-1], labels=fake_corr.columns[:-1], rotation=45)

    plt.subplot(1, 3, 3)
    ax = heatmap(diff_corr, cmap="coolwarm", annot=True, vmax=1, vmin=-1, cbar=False, mask=diff_mask)
    plt.title(f"Difference (abs)")
    plt.yticks([], labels=[])
    plt.xticks(ticks=ax.get_xticks()[:-1], labels=diff_corr.columns[:-1], rotation=45)

    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    if show:
        plt.show()


def detection_score(real: Dataloader,
                    fake: pd.DataFrame,
                    model: any,
                    n_folds: int = 5,
                    scoring: str = "roc_auc") -> tuple[float, float]:
    """
    Computes the detection score for the fake data.
    The detection score is the average score with which a classifier can distinguish between the real and the fake data.
    The score is computed using cross-validation with the given number of folds.
    Returns the average score and the standard deviation of the scores.

    Inputs:
    - real: Dataloader, the real data
    - fake: pd.DataFrame, the fake data
    - model: the classifier to use for the detection
    - n_folds: int, number of folds for the cross-validation, default=5
    - scoring: str, the scoring method from sklearn to use for the cross-validation, default="roc_auc"
    """
    cat_cols = real.cat_features.copy()
    target = real.target
    real = real.data
    # make sure we have the same number of real and fake examples
    if len(real) > len(fake):
        real = real.sample(n=len(fake), random_state=42)
    elif len(real) < len(fake):
        fake = fake.sample(n=len(real), random_state=42)
    
    real = real.copy(deep=True)
    fake = fake.copy(deep=True)

    # remove columns where only nan values
    for c in fake.columns:
        if fake[c].isna().all():
            fake = fake.drop(columns=[c], axis=1)
            real = real.drop(columns=[c], axis=1)
            if c in cat_cols:
                cat_cols.remove(c)

    real, encoder = one_hot_dataset(real, cat_cols + [target], encoder=None, encode_target=True, target_name=target)
    fake, _       = one_hot_dataset(fake, cat_cols + [target], encoder=encoder, encode_target=True, target_name=target)
    scaler = MinMaxScaler()
    real_scaled = scaler.fit_transform(real)
    fake_scaled = scaler.transform(fake)

    real = pd.DataFrame(real_scaled, columns=real.columns)
    fake = pd.DataFrame(fake_scaled, columns=fake.columns)

    # add labels and concat data
    real["fake"] = 0
    fake["fake"] = 1
    data = pd.concat([real, fake], ignore_index=True, axis=0)

    # cross val
    y = data["fake"]
    X = data.drop(columns=["fake"], axis=1)
    scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42), scoring=scoring)
    return scores.mean(), scores.std()


def ml_utility(real: Dataloader,
               fake: pd.DataFrame,
               model: BaseEstimator,
               n_folds: int = 5,
               test_size: float = 0.2,
               scoring: str = "roc_auc") -> tuple[float, float, float, float]:
    """
    Computes the machine learning utility of the fake data.
    Machine learning utility is the comparison between Train on Real, Test on Real (TRTR) and Train on Synthetic, Test on Real (TSTR).
    Returns a tuple with this order: (trtr_mean, trtr_std, tstr_mean, tstr_std)

    Inputs:
    - real: Dataloader, the real data
    - fake: pd.DataFrame, the fake data
    - model: BaseEstimator, the classifier from sklearn to use for the utility
    - n_folds: int, number of folds for the cross-validation, default=5
    - test_size: float, the size of the test set for the train-test split, default=0.2
    - scoring: str, the scoring method from sklearn to use for the cross-validation, default="roc_auc", possible value: ["roc_auc", "accuracy"]
    """
    if scoring not in ["roc_auc", "accuracy", "f1"]:
        raise ValueError(f"Scoring method {scoring} not recognized. Possible values are ['roc_auc', 'accuracy', 'f1']")
    
    target = real.target
    task = real.task
    pos_label = real.pos_label
    cat_cols = real.cat_features.copy()
    num_cols = real.num_features
    real = real.data

    if len(real) > len(fake):
        real = real.sample(n=len(fake), random_state=42)
    elif len(real) < len(fake):
        fake = fake.sample(n=len(real), random_state=42)

    real = real.copy(deep=True)
    fake = fake.copy(deep=True)

    # remove columns where only nan values
    for c in fake.columns:
        if fake[c].isna().all():
            fake = fake.drop(columns=[c], axis=1)
            real = real.drop(columns=[c], axis=1)
            if c in cat_cols:
                cat_cols.remove(c)

    # one hot dataset
    target_real = real[target].reset_index(drop=True)
    target_fake = fake[target].reset_index(drop=True)
    real = real.drop(columns=[target], axis=1)
    fake = fake.drop(columns=[target], axis=1)
    real, encoder = one_hot_dataset(real, cat_cols, encoder=None, encode_target=True, target_name=target)
    fake, _       = one_hot_dataset(fake, cat_cols, encoder=encoder, encode_target=True, target_name=target)
    scaler = MinMaxScaler()
    real = scaler.fit_transform(real)
    fake = scaler.transform(fake)

    cols = num_cols + [c for c in encoder.get_feature_names_out()]
    real = pd.DataFrame(real, columns=cols)
    fake = pd.DataFrame(fake, columns=cols)
    real[target] = target_real
    fake[target] = target_fake

    # if xbg, encode target
    if isinstance(model, XGBClassifier):
        label_enc = LabelEncoder()
        real[target] = label_enc.fit_transform(real[target])
        fake[target] = label_enc.transform(fake[target])
        pos_label = label_enc.transform([pos_label])[0]

    scores_trtr = []
    scores_tstr = []
    for i in range(n_folds):
        model_real = clone(model)
        model_fake = clone(model)

        df_real_train, df_real_test = train_test_split(real, test_size=test_size, shuffle=True, random_state=i, stratify=real[target])
        df_fake_train, _            = train_test_split(fake, test_size=test_size, shuffle=True, random_state=i, stratify=fake[target])

        y_train_real = df_real_train[target]
        X_train_real = df_real_train.drop(columns=[target], axis=1)
        y_test_real  = df_real_test[target]
        X_test_real  = df_real_test.drop(columns=[target], axis=1)
        y_train_fake = df_fake_train[target]
        X_train_fake = df_fake_train.drop(columns=[target], axis=1)

        model_real.fit(X_train_real, y_train_real)
        model_fake.fit(X_train_fake, y_train_fake)

        if task == "classification":
            if scoring == "roc_auc":
                pred_real = model_real.predict_proba(X_test_real)[:, 1]
                pred_fake = model_fake.predict_proba(X_test_real)[:, 1]
                scores_trtr.append(roc_auc_score(y_test_real, pred_real))
                scores_tstr.append(roc_auc_score(y_test_real, pred_fake))
            elif scoring == "accuracy" or scoring == "f1": 
                pred_real = model_real.predict(X_test_real)
                pred_fake = model_fake.predict(X_test_real)
                if scoring == "accuracy":
                    scores_trtr.append(accuracy_score(y_test_real, pred_real))
                    scores_tstr.append(accuracy_score(y_test_real, pred_fake))
                else:
                    scores_trtr.append(f1_score(y_test_real, pred_real, pos_label=pos_label))
                    scores_tstr.append(f1_score(y_test_real, pred_fake, pos_label=pos_label))
        else:
            pred_real = model_real.predict(X_test_real)
            pred_fake = model_fake.predict(X_test_real)
            scores_trtr.append(mean_squared_error(y_test_real, pred_real))
            scores_tstr.append(mean_squared_error(y_test_real, pred_fake))
    return np.mean(scores_trtr), np.std(scores_trtr), np.mean(scores_tstr), np.std(scores_tstr)


def ml_utility_combined(real: Dataloader,
                        fake: pd.DataFrame,
                        model: BaseEstimator,
                        n_folds: int = 5,
                        test_size: float = 0.2,
                        scoring: str = "roc_auc") -> tuple[float, float]:
    """
    Computes the machine learning utility of the fake data.
    Adds the fake data to the real data and computes the utility score.

    Inputs:
    - real: Dataloader, the real data
    - fake: pd.DataFrame, the fake data
    - model: BaseEstimator, the classifier from sklearn to use for the utility
    - n_folds: int, number of folds for the cross-validation, default=5
    - test_size: float, the size of the test set for the train-test split, default=0.2
    - scoring: str, the scoring method from sklearn to use for the cross-validation, default="roc_auc", possible value: ["roc_auc", "accuracy"]
    """
    if scoring not in ["roc_auc", "accuracy", "f1"]:
        raise ValueError(f"Scoring method {scoring} not recognized. Possible values are ['roc_auc', 'accuracy', 'f1']")
    
    target = real.target
    task = real.task
    pos_label = real.pos_label
    cat_cols = real.cat_features.copy()
    num_cols = real.num_features
    real = real.data

    real = real.copy(deep=True)
    fake = fake.copy(deep=True)

    # remove columns where only nan values
    for c in fake.columns:
        if fake[c].isna().all():
            fake = fake.drop(columns=[c], axis=1)
            real = real.drop(columns=[c], axis=1)
            if c in cat_cols:
                cat_cols.remove(c)

    # concat the real and the fake data
    df = pd.concat([real, fake], ignore_index=True, axis=0)
    df = df.sample(frac=1)

    # one hot dataset
    target_df = df[target].reset_index(drop=True)
    df = df.drop(columns=[target], axis=1)
    df, encoder = one_hot_dataset(df, cat_cols, encoder=None, encode_target=True, target_name=target)
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)

    cols = num_cols + [c for c in encoder.get_feature_names_out()]
    df = pd.DataFrame(df, columns=cols)
    df[target] = target_df

    # if xbg, encode target
    if isinstance(model, XGBClassifier):
        label_enc = LabelEncoder()
        df[target] = label_enc.fit_transform(df[target])
        pos_label = label_enc.transform([pos_label])[0]

    scores = []
    for i in range(n_folds):
        model_df = clone(model)

        df_train, df_test = train_test_split(df, test_size=test_size, shuffle=True, random_state=i, stratify=df[target])

        y_train = df_train[target]
        X_train = df_train.drop(columns=[target], axis=1)
        y_test  = df_test[target]
        X_test  = df_test.drop(columns=[target], axis=1)

        model_df.fit(X_train, y_train)

        if task == "classification":
            if scoring == "roc_auc":
                pred = model_df.predict_proba(X_test)[:, 1]
                scores.append(roc_auc_score(y_test, pred))
            elif scoring == "accuracy" or scoring == "f1":
                pred = model_df.predict(X_test)
                if scoring == "accuracy":
                    scores.append(accuracy_score(y_test, pred))
                else:
                    scores.append(f1_score(y_test, pred, pos_label=pos_label))
        else:
            pred = model_df.predict(X_test)
            scores.append(mean_squared_error(y_test, pred))
    return np.mean(scores), np.std(scores)


def prdc_OLD(real: Dataloader, fake: pd.DataFrame, nearest_k: int = 5) -> tuple[float, float, float, float]:
    """
    Computes the PRDC scores for the fake data over n_fols.
    PRDC: precison, recall, density, coverage
    Returns a tuple with the scores in this order: (precision, recall, density, coverage)

    Inputs:
    - real: Dataloader, the real data
    - fake: pd.DataFrame, the fake data
    - nearest_k: int, number of nearest neighbors to consider for the PRDC score, default=5
    """
    cat_feats = real.cat_features.copy()
    target = real.target
    real = real.data

    if len(real) > len(fake):
        real = real.sample(n=len(fake), random_state=42)
    elif len(real) < len(fake):
        fake = fake.sample(n=len(real), random_state=42)
    
    real = real.copy(deep=True)
    fake = fake.copy(deep=True)

    # remove columns where only nan values
    for c in fake.columns:
        if fake[c].isna().all():
            fake = fake.drop(columns=[c], axis=1)
            real = real.drop(columns=[c], axis=1)
            if c in cat_feats:
                cat_feats.remove(c)

    # encode data with one-hot and normalize them
    real, encoder = one_hot_dataset(real, cat_feats + [target], encoder=None, encode_target=True, target_name=target)
    fake, _       = one_hot_dataset(fake, cat_feats + [target], encoder=encoder, encode_target=True, target_name=target)
    scaler = MinMaxScaler()
    real = scaler.fit_transform(real)
    fake = scaler.transform(fake)


    real = GenericDataLoader(real)
    fake = GenericDataLoader(fake)

    prdc = eval_statistical.PRDCScore(nearest_k=nearest_k)
    scores = prdc.evaluate(real, fake)
    return scores["precision"], scores["recall"], scores["density"], scores["coverage"]


def prdc(real: Dataloader, fake: pd.DataFrame, nearest_k: int = 5, n_folds: int = 5, max_data: int = 350) -> dict:
    """
    Computes the PRDC scores for the fake data over n_fols.
    PRDC: precison, recall, density, coverage
    Returns a dictionary with the mean and std of the scores. {'precision': {'mean': ..., 'std': ...}, ...}

    Inputs:
    - real: Dataloader, the real data
    - fake: pd.DataFrame, the fake data
    - nearest_k: int, number of nearest neighbors to consider for the PRDC score, default=5
    - n_folds: int, number of times the prdc is evaluated. The dataset with highest number of examples is sampled each time with different seed. Default=5
    - max_data: int, maximum number of examples to consider for the PRDC score (due to high usage of RAM). Default=350
    """
    cat_feats = real.cat_features.copy()
    target = real.target
    real = real.data

    scores_prdc = {"precision": [], "recall": [], "density": [], "coverage": []}
    n_data_limit = min(max_data, len(real), len(fake))

    for i in range(n_folds):
        if len(real) > n_data_limit:
            real_sample = real.sample(n=n_data_limit, random_state=i)
        else:
            real_sample = real.copy(deep=True)
        if len(fake) > n_data_limit:
            fake_sample = fake.sample(n=n_data_limit, random_state=i)
        else:
            fake_sample = fake.copy(deep=True)

        # remove columns where only nan values
        for c in fake_sample.columns:
            if fake_sample[c].isna().all():
                fake_sample = fake_sample.drop(columns=[c], axis=1)
                real_sample = real_sample.drop(columns=[c], axis=1)
                if c in cat_feats:
                    cat_feats.remove(c)
        
        # encode data with one-hot and normalize them
        real_sample, encoder = one_hot_dataset(real_sample, cat_feats + [target], encoder=None, encode_target=True, target_name=target)
        fake_sample, _       = one_hot_dataset(fake_sample, cat_feats + [target], encoder=encoder, encode_target=True, target_name=target)
        scaler = MinMaxScaler()
        real_sample = scaler.fit_transform(real_sample)
        fake_sample = scaler.transform(fake_sample)

        real_sample = GenericDataLoader(real_sample)
        fake_sample = GenericDataLoader(fake_sample)

        prdc = eval_statistical.PRDCScore(nearest_k=nearest_k)
        scores = prdc.evaluate(real_sample, fake_sample)

        names_prdc = ["precision", "recall", "density", "coverage"]
        for name in names_prdc:
            scores_prdc[name].append(scores[name])

    all_scores = {}
    for name, lst in scores_prdc.items():
        all_scores[name] = {"mean": np.mean(lst), "std": np.std(lst)}

    return all_scores


def alpha_precision_beta_recall_OLD(real: Dataloader, fake: pd.DataFrame) -> tuple[float, float]:
    """
    Computes the alpha-precision and beta-recall scores for the fake data.
    Returns a tuple with scores: (alpha-precision, beta-recall)
    
    Inputs:
    - real: Dataloader, the real data
    - fake: pd.DataFrame, the fake data
    """
    cat_feats = real.cat_features.copy()
    target = real.target
    real = real.data

    if len(real) > len(fake):
        real = real.sample(n=len(fake), random_state=42)
    elif len(real) < len(fake):
        fake = fake.sample(n=len(real), random_state=42)

    real = real.copy(deep=True)
    fake = fake.copy(deep=True)

    # remove columns where only nan values
    for c in fake.columns:
        if fake[c].isna().all():
            fake = fake.drop(columns=[c], axis=1)
            real = real.drop(columns=[c], axis=1)
            if c in cat_feats:
                cat_feats.remove(c)

    # encode data with one-hot and normalize them
    real, encoder = one_hot_dataset(real, cat_feats + [target], encoder=None, encode_target=True, target_name=target)
    fake, _       = one_hot_dataset(fake, cat_feats + [target], encoder=encoder, encode_target=True, target_name=target)
    scaler = MinMaxScaler()
    real = scaler.fit_transform(real)
    fake = scaler.transform(fake)

    real = GenericDataLoader(real)
    fake = GenericDataLoader(fake)

    alpha_precision = eval_statistical.AlphaPrecision()
    scores = alpha_precision.evaluate(real, fake)
    return scores["delta_precision_alpha_OC"], scores["delta_coverage_beta_OC"]


def alpha_precision_beta_recall(real: Dataloader, fake: pd.DataFrame, max_data: int = 350, n_folds: int = 5) -> dict:
    """
    Computes the alpha-precision and beta-recall scores for the fake data.
    Returns a dict with the mean and std of the scores. {'alpha_precision': {'mean': ..., 'std': ...}, ...}
    
    Inputs:
    - real: Dataloader, the real data
    - fake: pd.DataFrame, the fake data
    - max_data: int, maximum number of examples to consider (due to high usage of RAM). Default=350
    - n_folds: int, number of times the precision and recall are evaluated. Default=5
    """
    cat_feats = real.cat_features.copy()
    target = real.target
    real = real.data

    n_data_limit = min(max_data, len(real), len(fake))
    scores_pr = {"alpha_precision": [], "beta_recall": []}

    for i in range(n_folds):
        if len(real) > n_data_limit:
            real_sample = real.sample(n=n_data_limit, random_state=i)
        else:
            real_sample = real.copy(deep=True)
        if len(fake) > n_data_limit:
            fake_sample = fake.sample(n=n_data_limit, random_state=i)
        else:
            fake_sample = fake.copy(deep=True)

        # remove columns where only nan values
        for c in fake_sample.columns:
            if fake_sample[c].isna().all():
                fake_sample = fake_sample.drop(columns=[c], axis=1)
                real_sample = real_sample.drop(columns=[c], axis=1)
                if c in cat_feats:
                    cat_feats.remove(c)
        
        # one hot encode the data and normalize them
        real_sample, encoder = one_hot_dataset(real_sample, cat_feats + [target], encoder=None, encode_target=True, target_name=target)
        fake_sample, _       = one_hot_dataset(fake_sample, cat_feats + [target], encoder=encoder, encode_target=True, target_name=target)
        scaler = MinMaxScaler()
        real_sample = scaler.fit_transform(real_sample)
        fake_sample = scaler.transform(fake_sample)

        real_sample = GenericDataLoader(real_sample)
        fake_sample = GenericDataLoader(fake_sample)

        alpha_precision = eval_statistical.AlphaPrecision()
        scores = alpha_precision.evaluate(real_sample, fake_sample)

        scores_pr["alpha_precision"].append(scores["delta_precision_alpha_OC"])
        scores_pr["beta_recall"].append(scores["delta_coverage_beta_OC"])
    
    all_scores = {}
    for name, lst in scores_pr.items():
        all_scores[name] = {"mean": np.mean(lst), "std": np.std(lst)}
    return all_scores


#################################################################################################

def print_results(res: dict, model_name: str):
    print(f"---MODEL NAME: {model_name}---")
    first_utility = True
    first_detection = True
    for key, value in res.items():
        if "utility" in key:
            if first_utility:
                first_utility = False
                print("Utility:")
            _, model_name, scoring, cat = key.split("-")
            scoring = "AUC" if scoring == "roc_auc" else scoring
            scoring = "ACC" if scoring == "accuracy" else scoring
            scoring = "F1" if scoring == "f1" else scoring
            mean_val, std_val = value["mean"], value["std"]
            print(f"\t{model_name:<16s} - {cat.upper()} - {scoring:>5s}: {mean_val:.3f}  ({std_val:.3f})")
        elif "detection" in key:
            if first_detection:
                first_detection = False
                print("Detection:")
            _, model_name, scoring = key.split("-")
            scoring = "AUC" if scoring == "roc_auc" else scoring
            scoring = "ACC" if scoring == "accuracy" else scoring
            scoring = "F1" if scoring == "f1" else scoring
            mean_val, std_val = value["mean"], value["std"]
            print(f"\t{model_name:<16s} - {scoring:>5s}: {mean_val:.3f}  ({std_val:.3f})")
        else:
            print(f"{key:<16s}: {value:.3f}")


def evaluate_synth_data(real: Dataloader,
                        fake: pd.DataFrame,
                        model_name: str,
                        n_folds: int = 5,
                        test_size: float = 0.2,
                        nearest_k:int = 5,
                        show_plots: bool = False,
                        verbose:bool = False) -> dict:
    """
    Launches all the evaluation functions

    Inputs:
    - real: Dataloader, the real data
    - fake: pd.DataFrame, the fake data
    - model_name: str, the name of the generative model used to obtain the fake data
    - n_folds: int, number of folds for the cross-validation, default=5
    - test_size: float, the size of the test set to use in the cross validation for the utility score, default=0.2
    - nearest_k: int, number of nearest neighbors to consider for the PRDC score, default=5
    - show_plots: bool, whether to show the plots or not, default=False
    - verbose: bool, whether to show the print statements or not, default=False
    """
    all_scores = {}

    dname = real.dname
    path = f"plots/{dname}/"
    save_format = "pdf"
    os.makedirs(path, exist_ok=True)
    gen_model_name = model_name

    classif_models = {
        "DT": DecisionTreeClassifier,
        "RF": RandomForestClassifier,
        "XGB": XGBClassifier
    }
    scorings = ["roc_auc", "accuracy", "f1"]

    # distribution plots
    plot_distributions(real, fake, show=show_plots, path=f"{path}distributions_real_fake_{gen_model_name}.{save_format}")
    if verbose:
        print(f"Distributions plot saved at {path}distributions_real_fake_{gen_model_name}.{save_format}")

    # PCA
    # soon

    # correlation matrix
    plot_correlation_matrix(real, fake, show=show_plots, path=f"{path}correlation_matrix_real_fake_{gen_model_name}.{save_format}")
    if verbose:
        print(f"Correlation matrix plot saved at {path}correlation_matrix_real_fake_{gen_model_name}.{save_format}")

    # use the same data for all the scores
    # if len(real.data) > len(fake):
    #     real.data = real.data.sample(n=len(fake), random_state=42)
    # elif len(real.data) < len(fake):
    #     fake = fake.sample(n=len(real.data), random_state=42)

    # detection score
    for model_name, model in classif_models.items():
        for scoring in scorings:
            mean_score, std_score = detection_score(real, fake, model(), n_folds=n_folds, scoring=scoring)
            all_scores[f"detection-{model_name}-{scoring}"] = {"mean": mean_score, "std": std_score}       

    # ml utility
    for model_name, model in classif_models.items():
        for scoring in scorings:
            # check that we have more than one class in the target
            if len(fake[real.target].unique()) <= 1:
                trtr_mean, trtr_std, tstr_mean, tstr_std = 0, 0, 0, 0
            else:
                trtr_mean, trtr_std, tstr_mean, tstr_std = ml_utility(real, fake, model(), n_folds=n_folds, test_size=test_size, scoring=scoring)
            all_scores[f"utility-{model_name}-{scoring}-trtr"] = {"mean": trtr_mean, "std": trtr_std}
            all_scores[f"utility-{model_name}-{scoring}-tstr"] = {"mean": tstr_mean, "std": tstr_std}

    for model_name, model in classif_models.items():
        for scoring in scorings:
            # check that we have more than one class in the target
            if len(fake[real.target].unique()) <= 1:
                mean_score, std_score = 0, 0
            else:
                mean_score, std_score = ml_utility_combined(real, fake, model(), n_folds=n_folds, test_size=test_size, scoring=scoring)
            all_scores[f"utility-{model_name}-{scoring}-combined"] = {"mean": mean_score, "std": std_score}

    # prdc
    prdc_scores = prdc(real, fake, nearest_k=nearest_k, n_folds=n_folds, max_data=350)
    for key, value in prdc_scores.items():
        all_scores[key] = value

    # alpha-precision and beta-recall
    alpha_beta_scores = alpha_precision_beta_recall(real, fake, max_data=350, n_folds=n_folds)
    for key, value in alpha_beta_scores.items():
        all_scores[key] = value

    # # prdc
    # prec, recall, density, coverage = prdc(real, fake, nearest_k=nearest_k)
    # all_scores["precision"] = prec
    # all_scores["recall"] = recall
    # all_scores["density"] = density
    # all_scores["coverage"] = coverage

    # # alpha-precision and beta-recall
    # alpha_rec, beta_recall = alpha_precision_beta_recall(real, fake)
    # all_scores["alpha_precision"] = alpha_rec
    # all_scores["beta_recall"] = beta_recall

    if verbose:
        print_results(all_scores, gen_model_name)

    return all_scores


def eval_all_models(dname):
    models = ["ctgan", "epic", "great", "tabddpm", "tabula", "synthloop"]

    df = pd.read_csv(f"synthloop/generated_examples/gen_data_{dname}_llama3.1_3_iters_sentence_synthloop.csv")
    print(f"full size: {len(df)}, no dup: {len(df.drop_duplicates())}, diff: {len(df) - len(df.drop_duplicates())}")
    LIMIT_SIZE_FAKE = len(df.drop_duplicates())
    print(f"LIMIT SIZE: {LIMIT_SIZE_FAKE}")

    all_scores = []
    for model_name in models:
        real = Dataloader(dname, parent_directory="synthloop/data")
        if model_name == "synthloop":
            fake = pd.read_csv(f"synthloop/generated_examples/gen_data_{dname}_llama3.1_3_iters_sentence_synthloop.csv")
        else:
            fake = pd.read_csv(f"synthloop/generated_examples/{model_name}_gen_data_{dname}.csv")

        n_dup = len(fake) - len(fake.drop_duplicates())
        print(f"{model_name}: size fake: {len(fake)}, n duplicates: {len(fake) - len(fake.drop_duplicates())}, size no dup: {len(fake.drop_duplicates())}")
        fake = fake.drop_duplicates()
        fake = fake.sample(n=LIMIT_SIZE_FAKE, random_state=42)

        scores = evaluate_synth_data(real, fake, model_name, n_folds=5, test_size=0.2, nearest_k=5, show_plots=False, verbose=True)
        scores["Duplicates"] = n_dup
        scores["model_name"] = model_name
        all_scores.append(scores)
    
    df_scores = pd.DataFrame(all_scores)
    df_scores.to_csv(f"{dname}_results.csv", index=False)


def eval_iterations_synthloop(dname, n_iter=3, n_round=7):
    # merge data
    # df_full = pd.DataFrame()
    # for i in range(n_iter):
    #     df_iter = pd.DataFrame()
    #     for j in range(n_round):
    #         try:
    #             df = pd.read_csv(f"synthloop/generated_examples/full_test_{dname}/gen_data_{dname}_{j}_iter_{i}_llama3.1_sentence.csv")
    #         except pd.errors.EmptyDataError:
    #             print(f"skip {i} {j}")
    #             continue
    #         print(f"size df {i} {j}: {len(df)}")
    #         df_iter = pd.concat([df_iter, df], axis=0, ignore_index=True)
    #     print(f"total size iter {i}: {len(df_iter)}")
    #     df_iter["iter"] = i
    #     df_full = pd.concat([df_full, df_iter], axis=0, ignore_index=True)
    # print(f"total size: {len(df_full)}")
    # df_full.to_csv(f"synthloop/generated_examples/gen_data_{dname}_all_sentence.csv", index=False)


    # df = pd.read_csv(f"synthloop/generated_examples/gen_data_{dname}_all_iters.csv")
    # df_2 = pd.read_csv(f"synthloop/generated_examples/gen_data_{dname}_all_iters_1.csv")
    # df = pd.concat([df, df_2], axis=0, ignore_index=True)

    df = pd.read_csv(f"synthloop/generated_examples/gen_data_{dname}_all_sentence.csv")
    N_LIMIT = 1e6
    for i in range(n_iter):
        df_iter = df[df["iter"] == i]
        print(f"iter {i}: {len(df_iter)}, no dup: {len(df_iter.drop_duplicates())}, diff: {len(df_iter) - len(df_iter.drop_duplicates())}")
        N_LIMIT = min(N_LIMIT, len(df_iter.drop_duplicates()))
    print(f"LIMIT: {N_LIMIT}")

    all_scores = []
    for i in range(n_iter):
        df_iter = df[df["iter"] == i]
        if False:
            fake = pd.read_csv("synthloop/generated_examples/synthloop_gen_data_adult.csv")
        else:
            fake = df_iter.drop(columns=["iter"], axis=1)
        real = Dataloader(dname=dname, parent_directory="synthloop/data")
        # count number of duplicate rows
        print(f"Original fake length: {len(fake)}, no dup length: {len(fake.drop_duplicates())}, diff: {len(fake) - len(fake.drop_duplicates())}")
        fake = fake.drop_duplicates()
        fake = fake.sample(n=N_LIMIT, random_state=42)
        scores = evaluate_synth_data(real, fake, f"synthloop_{i}", n_folds=5, test_size=0.2, nearest_k=5, show_plots=False, verbose=True)
        scores["model_name"] = f"synthloop_{i}"
        all_scores.append(scores)
    df_scores = pd.DataFrame(all_scores)
    df_scores.to_csv(f"{dname}_results_synthloop_sentence.csv", index=False)
    

def eval_dataframes(dname, fake_frames, fake_names, filename):
    # compute limit size of data due to duplicates
    LIMIT_SIZE_FAKE = 1e6
    model_min = None
    real_df = Dataloader(dname, parent_directory="data")
    for i, fake in enumerate(fake_frames):

        # remove the duplicates and the number of collisions
        df_fake = fake.drop_duplicates()
        df_full = pd.concat([real_df.data.drop_duplicates(), df_fake], ignore_index=True)
        n_coll = (len(real_df.data.drop_duplicates()) + len(df_fake)) - len(df_full.drop_duplicates())

        actual_size = len(df_fake) - n_coll

        if actual_size < LIMIT_SIZE_FAKE:
            LIMIT_SIZE_FAKE = actual_size
            model_min = fake_names[i]
        # print(f"{fake_names[i]}: {len(fake.drop_duplicates())}")
    print(f"LIMIT SIZE: {LIMIT_SIZE_FAKE}, model min: {model_min}")

    all_scores = []
    for model_name, fake in zip(fake_names, fake_frames):
        real = Dataloader(dname, parent_directory="data")
        n_dup = len(fake) - len(fake.drop_duplicates())
        og_fake_size = len(fake)
        no_dup_fake_size = len(fake.drop_duplicates())
        
        # remove the duplicates
        fake = fake.drop_duplicates()

        # remove the collisions with the real data
        df_real = real.data.copy(deep=True).drop_duplicates()
        df_fake = fake.copy(deep=True)
        df_real["fake"] = 0
        df_fake["fake"] = 1
        df_full = pd.concat([df_real, df_fake], axis=0, ignore_index=True)
        n_collisions = len(df_full) - len(df_full.drop_duplicates(subset=df_real.columns[:-1]))
        df_full = df_full.drop_duplicates(subset=df_real.columns[:-1])
        fake = df_full[df_full["fake"] == 1].drop(columns=["fake"], axis=1)

        # sample data to the minimum size
        # real.data = real.data.sample(n=LIMIT_SIZE_FAKE, random_state=42) # -> done in each of the functions
        fake = fake.sample(n=LIMIT_SIZE_FAKE, random_state=42)
        scores = evaluate_synth_data(real, fake, model_name, n_folds=5, test_size=0.2, nearest_k=5, show_plots=False, verbose=False)
        scores["Duplicates"] = n_dup/og_fake_size *100
        scores["Collisions"] = n_collisions/no_dup_fake_size *100
        scores["model_name"] = model_name
        all_scores.append(scores)
    df_scores = pd.DataFrame(all_scores)
    df_scores.to_csv(filename, index=False)


def merge_dataset_iterations(folder, dname, name_template, n_iter, n_round):
    df_full = pd.DataFrame()
    for i in range(n_iter):
        df_iter = pd.DataFrame()
        for j in range(n_round):
            try:
                filename = name_template.format(dname=dname, i_round=j, i_iter=i)
                df = pd.read_csv(f"{folder}/{filename}.csv")
            except pd.errors.EmptyDataError:
                # print(f"skip {i} {j}")
                continue
            # print(f"size df {i} {j}: {len(df)}")
            df_iter = pd.concat([df_iter, df], axis=0, ignore_index=True)
        # print(f"total size iter {i}: {len(df_iter)}")
        df_iter["iter"] = i
        df_full = pd.concat([df_full, df_iter], axis=0, ignore_index=True)
    return df_full




if __name__ == "__main__":

    ########################################################################
    ####################     COMPETITORS COMPARISON     ####################
    ########################################################################

    print("running competitors comparison...", flush=True)

    ## adult
    dname = "adult"
    fake_frames = []
    fake_names = ["statgen", "ctgan", "epic", "great", "tabddpm", "tabula", "synthloop"]
    for name in fake_names[:-1]:
        fake_frames.append(pd.read_csv(f"generated_examples/{name}_gen_data_{dname}.csv"))
    fake_frames.append(pd.read_csv(f"generated_examples/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyFalse.csv"))
    eval_dataframes(dname, fake_frames, fake_names, f"evaluation_gen_data/{dname}/competitors_comparison.csv")
    print(f"saved as: evaluation_gen_data/{dname}/competitors_comparison.csv")

    ## thyroid
    dname = "thyroid"
    fake_frames = []
    fake_names = ["statgen", "ctgan", "epic", "great", "tabddpm", "tabula", "synthloop"]
    for name in fake_names[:-1]:
        fake_frames.append(pd.read_csv(f"generated_examples/{name}_gen_data_{dname}.csv"))
    fake_frames.append(pd.read_csv(f"generated_examples/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyFalse.csv"))
    eval_dataframes(dname, fake_frames, fake_names, f"evaluation_gen_data/{dname}/competitors_comparison.csv")
    print(f"saved as: evaluation_gen_data/{dname}/competitors_comparison.csv")




    #################################################################
    ####################     ITER COMPARISON     ####################
    #################################################################

    print("running iterations comparison...", flush=True)

    ## adult info-full_feedback
    dname = "adult"
    df_iters = merge_dataset_iterations(f"generated_examples/full_test_{dname}_csv_2",
                                        dname,
                                        "gen_data_{dname}_{i_round}_iter_{i_iter}_llama3.1_csv_datainfo_True",
                                        n_iter=3,
                                        n_round=5)
    fake_names = [f"iter_{i}" for i in range(3)]
    fake_frames = [df_iters[df_iters["iter"] == i].drop(columns=["iter"], axis=1) for i in range(3)]
    eval_dataframes(dname, fake_frames, fake_names, f"evaluation_gen_data/{dname}/iter_comparison_csv_info.csv")
    print(f"saved as: evaluation_gen_data/{dname}/iter_comparison_csv_info.csv")

    ## thyroid info-full_feedback
    dname = "thyroid"
    df_iters = merge_dataset_iterations(f"generated_examples/full_test_{dname}_csv_info",
                                        dname,
                                        "gen_data_{dname}_{i_round}_iter_{i_iter}_llama3.1_csv_noInfoFalse_weaknessOnlyFalse",
                                        n_iter=3,
                                        n_round=7)
    fake_names = [f"iter_{i}" for i in range(3)]
    fake_frames = [df_iters[df_iters["iter"] == i].drop(columns=["iter"], axis=1) for i in range(3)]
    eval_dataframes(dname, fake_frames, fake_names, f"evaluation_gen_data/{dname}/iter_comparison_csv_info.csv")
    print(f"saved as: evaluation_gen_data/{dname}/iter_comparison_csv_info.csv")




    #####################################################################
    ####################     VARIANTS COMPARISON     ####################
    #####################################################################

    print("running variants comparison...", flush=True)

    ## adult
    dname = "adult"
    fake_frames = []
    fake_names = ["baseline",
                  "temp. 0.9",
                  "30 shots",
                  "cat first",
                  "num first",
                  "EPIC",
                  "fshots feedback",
                  "deepseek-r1"]
    folder = "generated_examples"
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_temp0.9.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_30shots.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_cat_first.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_num_first.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_epicFewShots.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_shotFeedback.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_deepseekr1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue.csv"))

    eval_dataframes(dname, fake_frames, fake_names, f"evaluation_gen_data/{dname}/variants_synthloop_csv_comparison.csv")
    print(f"saved as: evaluation_gen_data/{dname}/variants_synthloop_csv_comparison.csv")




    ##########################################################################
    ####################     INFO-WEAKNESS COMPARISON     ####################
    ##########################################################################

    print("running info-weakness comparison...", flush=True)
    
    ## adult, csv, temp 0.7, shots 20, no shots feedback
    dname = "adult"
    fake_frames = []
    fake_names = ["info-full_fb", "info-only_weakness", "no_info-full_fb", "no_info-only_weakness"]
    folder = "generated_examples"
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyFalse.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyTrue.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyFalse.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue.csv"))

    eval_dataframes(dname, fake_frames, fake_names, f"evaluation_gen_data/{dname}/info_weakness_csv_comparison.csv")
    print(f"saved as: evaluation_gen_data/{dname}/info_weakness_csv_comparison.csv")


    ## thyroid, csv, temp 0.7, shots 20, no shots feedback
    dname = "thyroid"
    fake_frames = []
    fake_names = ["info-full_fb", "info-only_weakness", "no_info-full_fb", "no_info-only_weakness"]
    folder = "generated_examples"
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyFalse.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyTrue.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyFalse.csv"))
    fake_frames.append(pd.read_csv(f"{folder}/gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue.csv"))

    eval_dataframes(dname, fake_frames, fake_names, f"evaluation_gen_data/{dname}/info_weakness_csv_comparison.csv")
    print(f"saved as: evaluation_gen_data/{dname}/info_weakness_csv_comparison.csv")


    ## adult, csv, temp 0.9, shots 30
    dname = "adult"
    fake_frames = []
    fake_names = ["info-full",
                  "info-full-fs_fb",
                  "info-weakness",
                  "info-weakness-fs_fb",
                  "no_info-full",
                  "no_info-full-fs_fb",
                  "no_info-weakness",
                  "no_info-weakness-fs_fb"]
    folder = "generated_examples"
    files = [f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyFalse_temp0.9_shots30",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyFalse_temp0.9_shots30_shotFbTrue",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyTrue_temp0.9_shots30",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyTrue_temp0.9_shots30_shotFbTrue",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyFalse_temp0.9_shots30",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyFalse_temp0.9_shots30_shotFbTrue",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_temp0.9_shots30",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_temp0.9_shots30_shotFbTrue"]
    for file in files:
        fake_frames.append(pd.read_csv(f"{folder}/{file}.csv"))
    
    eval_dataframes(dname, fake_frames, fake_names, f"evaluation_gen_data/{dname}/info_weakness_csv_temp0.9_shots30_comparison.csv")
    print(f"saved as: evaluation_gen_data/{dname}/info_weakness_csv_temp0.9_shots30_comparison.csv")


    ## thyroid, csv, temp 0.9, shots 30
    dname = "thyroid"
    fake_frames = []
    fake_names = ["info-full",
                  "info-full-fs_fb",
                  "info-weakness",
                  "info-weakness-fs_fb",
                  "no_info-full",
                  "no_info-full-fs_fb",
                  "no_info-weakness",
                  "no_info-weakness-fs_fb"]
    folder = "generated_examples"
    files = [f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyFalse_temp0.9_shots30_shotFbFalse",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyFalse_temp0.9_shots30_shotFbTrue",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyTrue_temp0.9_shots30_shotFbFalse",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyTrue_temp0.9_shots30_shotFbTrue",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyFalse_temp0.9_shots30_shotFbFalse",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyFalse_temp0.9_shots30_shotFbTrue",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_temp0.9_shots30_shotFbFalse",
             f"gen_data_{dname}_llama3.1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_temp0.9_shots30_shotFbTrue"]
    for file in files:
        fake_frames.append(pd.read_csv(f"{folder}/{file}.csv"))
    
    eval_dataframes(dname, fake_frames, fake_names, f"evaluation_gen_data/{dname}/info_weakness_csv_temp0.9_shots30_comparison.csv")
    print(f"saved as: evaluation_gen_data/{dname}/info_weakness_csv_temp0.9_shots30_comparison.csv")


    ## adult, csv, temp 0.9, shots 30, deepseek-r1
    dname = "adult"
    fake_frames = []
    fake_names = ["info-full",
                  "info-full-fs_fb",
                  "info-weakness",
                  "info-weakness-fs_fb",
                  "no_info-full",
                  "no_info-full-fs_fb",
                  "no_info-weakness",
                  "no_info-weakness-fs_fb"]
    folder = "generated_examples"
    files = [f"gen_data_{dname}_deepseekr1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyFalse_temp0.9_shots30_shotFbFalse",
             f"gen_data_{dname}_deepseekr1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyFalse_temp0.9_shots30_shotFbTrue",
             f"gen_data_{dname}_deepseekr1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyTrue_temp0.9_shots30_shotFbFalse",
             f"gen_data_{dname}_deepseekr1_3_iters_csv_synthloop_noInfoFalse_weaknessOnlyTrue_temp0.9_shots30_shotFbTrue",
             f"gen_data_{dname}_deepseekr1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyFalse_temp0.9_shots30_shotFbFalse",
             f"gen_data_{dname}_deepseekr1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyFalse_temp0.9_shots30_shotFbTrue",
             f"gen_data_{dname}_deepseekr1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_temp0.9_shots30_shotFbFalse",
             f"gen_data_{dname}_deepseekr1_3_iters_csv_synthloop_noInfoTrue_weaknessOnlyTrue_temp0.9_shots30_shotFbTrue"]
    for file in files:
        fake_frames.append(pd.read_csv(f"{folder}/{file}.csv"))
    
    eval_dataframes("adult", fake_frames, fake_names, f"evaluation_gen_data/{dname}/info_weakness_csv_deepseek_temp0.9_shots30_comparison.csv")
    print(f"saved as: evaluation_gen_data/{dname}/info_weakness_csv_deepseek_temp0.9_shots30_comparison.csv")




