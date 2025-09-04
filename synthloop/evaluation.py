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
import lightgbm as lgb

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


def get_df_without_collisions_duplicates(real: Dataloader,
                              fake: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Removes all the collisions and duplicates from the fake dataset and returns the resulting dataset and the number of collisions
    """
    df_real = real.data_full.copy(deep=True).drop_duplicates()
    df_fake = fake.copy(deep=True).drop_duplicates()
    df_real["fake"] = 0
    df_fake["fake"] = 1
    df_full = pd.concat([df_real, df_fake], axis=0, ignore_index=True)
    n_collisions = len(df_full) - len(df_full.drop_duplicates(subset=df_real.columns[:-1]))
    df_full = df_full.drop_duplicates(subset=df_real.columns[:-1])
    fake = df_full[df_full["fake"] == 1].drop(columns=["fake"], axis=1)
    return fake, n_collisions


def get_df_no_collision(real: Dataloader,
                        fake: pd.DataFrame) -> pd.DataFrame:
    df_real = real.data_full.copy(deep=True)
    df_fake = fake.copy(deep=True)
    for col in df_real.columns:
        if col not in df_fake.columns:
            df_real = df_real.drop(columns=[col], axis=1)
            # print(f"removed column {col} from real", flush=True)

    df_fake = df_fake.dropna()

    for col in df_real.columns:
        # change type of columns in fake to match the type of columns in real
        df_fake[col] = df_fake[col].astype(df_real[col].dtype)

    merged = df_fake.merge(df_real, on=list(df_real.columns), how="left", indicator=True)
    df_fake_filtered = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"], axis=1)
    return df_fake_filtered


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
    real = real.data_test

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
    real = real.data_test

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


def detection_score(dataloader: Dataloader,
                    train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    model: any,
                    n_folds: int = 5,
                    scoring: str = "roc_auc") -> tuple[float, float]:
    """
    Computes the detection score for the fake data.
    The detection score is the average score with which a classifier can distinguish between the real and the fake data.
    The score is computed using cross-validation with the given number of folds.
    Returns the average score and the standard deviation of the scores.

    Inputs:
    - dataloader: Dataloader, the real data and all their information
    - train_df: pd.DataFrame, the df on which the model will be trained on
    - test_df: pd.DataFrame, the df on which the model will be tested on. Must have a column "fake"
    - model: the classifier to use for the detection
    - n_folds: int, number of folds for the cross-validation, default=5
    - scoring: str, the scoring method from sklearn to use for the cross-validation, default="roc_auc"
    """
    if "fake" not in test_df.columns:
        raise ValueError("The test_df must have a column 'fake' to distinguish between the real and the fake data")
    if scoring not in ["roc_auc", "accuracy", "f1"]:
        raise ValueError(f"Scoring method {scoring} not recognized. Possible values are ['roc_auc', 'accuracy', 'f1']")
    
    target = dataloader.target

    train_df = train_df.copy(deep=True)
    test_df = test_df.copy(deep=True)

    # one hot encode and scale the data
    cat_feats = [f for f in dataloader.cat_features if f in train_df.columns]
    train_df, encoder = one_hot_dataset(train_df, cat_feats + [target], encoder=None, encode_target=True, target_name=target)
    test_df, _        = one_hot_dataset(test_df, cat_feats + [target], encoder=encoder, encode_target=True, target_name=target)
    scaler = MinMaxScaler()
    train_df_scaled = scaler.fit_transform(train_df)
    test_df_scaled = scaler.transform(test_df)
    train_df = pd.DataFrame(train_df_scaled, columns=train_df.columns)
    test_df = pd.DataFrame(test_df_scaled, columns=test_df.columns)

    # cross val
    scores = []
    for i in range(n_folds):
        model_pred = clone(model)
        X_train, y_train = train_df.drop(columns=["fake"], axis=1), train_df["fake"]
        X_test, y_test = test_df.drop(columns=["fake"], axis=1), test_df["fake"]

        if isinstance(model_pred, lgb.LGBMClassifier):
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test)
            params = {"objective": "binary", "verbose": -1, "num_threads": 1}
            model_pred = lgb.train(params, train_data, num_boost_round=100)
            y_pred = model_pred.predict(X_test)
            if scoring != "roc_auc":
                y_pred = (y_pred > 0.5).astype(int)
        else:
            model_pred.fit(X_train, y_train)
            if scoring == "roc_auc":
                y_pred = model_pred.predict_proba(X_test)[:, 1]
            else:
                y_pred = model_pred.predict(X_test)
        
        if scoring == "roc_auc":
            scores.append(roc_auc_score(y_test, y_pred))
        elif scoring == "accuracy":
            scores.append(accuracy_score(y_test, y_pred))
        # else:
        #     scores.append(f1_score(y_test, y_pred, pos_label=pos_label))
    return np.mean(scores), np.std(scores)


def ml_utility(dataloader: Dataloader,
               train_df: pd.DataFrame,
               model: any,
               n_folds: int = 5,
               scoring: str = "roc_auc") -> tuple[float, float]:
    """
    Computes the machine learning utility of the fake data.
    Machine learning utility is the comparison between Train on Real, Test on Real (TRTR) and Train on Synthetic, Test on Real (TSTR).
    Returns a tuple with this order: (trtr_mean, trtr_std, tstr_mean, tstr_std)

    Inputs:
    - dataloader: Dataloader, the real data with all their information
    - train_df: pd.DataFrame, the df on which the model will be trained on
    - model: BaseEstimator, the classifier from sklearn to use for the utility
    - n_folds: int, number of folds for the cross-validation, default=5
    - test_size: float, the size of the test set for the train-test split, default=0.2
    - scoring: str, the scoring method from sklearn to use for the cross-validation, default="roc_auc", possible value: ["roc_auc", "accuracy"]
    """
    if scoring not in ["roc_auc", "accuracy", "f1"]:
        raise ValueError(f"Scoring method {scoring} not recognized. Possible values are ['roc_auc', 'accuracy', 'f1']")
    
    target = dataloader.target
    train_df = train_df.copy(deep=True)
    test_df = dataloader.data_test.copy(deep=True)

    # remove columns from real test that are not in train (if any)
    for c in test_df.columns:
        if c not in train_df.columns:
            test_df = test_df.drop(columns=[c], axis=1)

    # one hot encode and scale the data
    train_target = train_df[target].reset_index(drop=True)
    test_target = test_df[target].reset_index(drop=True)
    train_df = train_df.drop(columns=[target], axis=1)
    test_df = test_df.drop(columns=[target], axis=1)

    cat_feats = [f for f in dataloader.cat_features if f in train_df.columns]
    train_df, encoder = one_hot_dataset(train_df, cat_feats, encoder=None, encode_target=True, target_name=target)
    test_df, _        = one_hot_dataset(test_df, cat_feats, encoder=encoder, encode_target=True, target_name=target)
    scaler = MinMaxScaler()
    train_df_scaled = scaler.fit_transform(train_df)
    test_df_scaled = scaler.transform(test_df)
    train_df = pd.DataFrame(train_df_scaled, columns=train_df.columns)
    test_df = pd.DataFrame(test_df_scaled, columns=test_df.columns)
    train_df[target] = train_target
    test_df[target] = test_target

    # if xgb, encode target
    if isinstance(model, XGBClassifier) or isinstance(model, lgb.LGBMClassifier):
        label_enc = LabelEncoder()
        train_df[target] = label_enc.fit_transform(train_df[target])
        test_df[target] = label_enc.transform(test_df[target])
        pos_label = label_enc.transform([dataloader.pos_label])[0]

    scores = []
    for i in range(n_folds):
        model_pred = clone(model)
        X_train, y_train = train_df.drop(columns=[target], axis=1), train_df[target]
        X_test, y_test = test_df.drop(columns=[target], axis=1), test_df[target]

        if isinstance(model_pred, lgb.LGBMClassifier):
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test)
            params = {"objective": "binary", "verbose": -1, "num_threads": 1}
            model_pred = lgb.train(params, train_data, num_boost_round=100)
            y_pred = model_pred.predict(X_test)
            if scoring != "roc_auc":
                y_pred = (y_pred > 0.5).astype(int)
        else:
            model_pred.fit(X_train, y_train)
            if scoring == "roc_auc":
                y_pred = model_pred.predict_proba(X_test)[:, 1]
            else:
                y_pred = model_pred.predict(X_test)
        
        if scoring == "roc_auc":
            scores.append(roc_auc_score(y_test, y_pred))
        elif scoring == "accuracy":
            scores.append(accuracy_score(y_test, y_pred))
        else:
            scores.append(f1_score(y_test, y_pred, pos_label=pos_label))
    return np.mean(scores), np.std(scores)


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

    # remove columns where only nan values
    for c in fake.columns:
        if fake[c].isna().all():
            fake = fake.drop(columns=[c], axis=1)

    # remove collisions from fake
    # fake, _ = get_df_without_collisions_duplicates(real, fake)
    fake = get_df_no_collision(real, fake)

    real = real.data # data_test may be too small to get a good estimate of the PRDC score

    scores_prdc = {"precision": [], "recall": [], "density": [], "coverage": []}

    fake = fake.drop_duplicates()
    limit_data = min(max_data, len(real), len(fake))

    for i in range(n_folds):
        real_sample = real.sample(n=limit_data, random_state=i)
        fake_sample = fake.sample(n=limit_data, random_state=i)

        # remove columns where only nan values
        for c in real_sample.columns:
            if c not in fake_sample.columns:
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
                        test_name: str,
                        n_data_limit: int,  # size of the smallest generated dataset - collisions
                        n_folds: int = 5,
                        test_size: float = 0.2,
                        nearest_k:int = 5,
                        show_plots: bool = False,
                        verbose:bool = False,) -> dict:
    """
    Launches all the evaluation functions

    Inputs:
    - real: Dataloader, the real data
    - fake: pd.DataFrame, the fake data
    - model_name: str, the name of the generative model used to obtain the fake data
    - test_name: str, the name of the test being run (used to save the plots)
    - n_data_limit: int, maximum number of examples to consider for the evaluation
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
        # "DT": DecisionTreeClassifier,
        "LGB": lgb.LGBMClassifier,
        "RF": RandomForestClassifier,
        "XGB": XGBClassifier
    }
    # scorings = ["roc_auc", "accuracy", "f1"]
    scorings = ["roc_auc"]


    ##########################
    ### distribution plots ###
    ##########################
    print("running distribution plots", flush=True)
    plot_distributions(real, fake, show=show_plots, path=f"{path}{test_name}_distributions_real_fake_{gen_model_name}.{save_format}")
    if verbose:
        print(f"Distributions plot saved at {path}{test_name}_distributions_real_fake_{gen_model_name}.{save_format}")


    ##########################
    ### correlation matrix ###
    ##########################
    print("running correlation matrix", flush=True)
    plot_correlation_matrix(real, fake, show=show_plots, path=f"{path}{test_name}_correlation_matrix_real_fake_{gen_model_name}.{save_format}")
    if verbose:
        print(f"Correlation matrix plot saved at {path}{test_name}_correlation_matrix_real_fake_{gen_model_name}.{save_format}")


    #######################
    ### detection score ###
    #######################
    print("running detection score", flush=True)
    # copy the train and test datasets
    test_split       = real.data_test.copy(deep=True)
    train_split_real = real.data.copy(deep=True)
    fake_data        = fake.copy(deep=True)

    # remove the columns where all the values are nan
    for c in fake_data.columns:
        if fake_data[c].isna().all():
            fake_data = fake_data.drop(columns=[c], axis=1)
            train_split_real = train_split_real.drop(columns=[c], axis=1)
            test_split = test_split.drop(columns=[c], axis=1)

    # remove the duplicates and the number of collisions
    # fake_data, _ = get_df_without_collisions_duplicates(real, fake_data)
    fake_data = get_df_no_collision(real, fake_data)

    # create the train and test datasets with correct number of data
    # downsample the sythetic data to the minimum size
    fake_data = fake_data.sample(n=n_data_limit, random_state=42)
    train_fake, test_fake = train_test_split(fake_data, test_size=0.2, random_state=42)

    if len(test_split) > len(test_fake):
        test_split = test_split.sample(n=len(test_fake), random_state=42)
    else:
        test_fake = test_fake.sample(n=len(test_split), random_state=42)

    if len(train_split_real) > len(train_fake):
        train_split_real = train_split_real.sample(n=len(train_fake), random_state=42)
    else:
        train_fake = train_fake.sample(n=len(train_split_real), random_state=42)

    train_fake["fake"] = 1
    test_fake["fake"] = 1
    train_split_real["fake"] = 0
    test_split["fake"] = 0

    train_df = pd.concat([train_split_real, train_fake], axis=0, ignore_index=True)
    test_df = pd.concat([test_split, test_fake], axis=0, ignore_index=True)

    for model_name, model in classif_models.items():
        for scoring in scorings:
            mean_score, std_score = detection_score(real, train_df, test_df, model(), n_folds=n_folds, scoring=scoring)
            all_scores[f"detection-{model_name}-{scoring}"] = {"mean": mean_score, "std": std_score}       

    ##################
    ### ml utility ###
    ##################
    print("running ml utility", flush=True)

    ### TRTR
    train_split_real = real.data.copy(deep=True)
    train_split_real = train_split_real.sample(n=n_data_limit, random_state=42)

    for model_name, model in classif_models.items():
        for scoring in scorings:
            trtr_mean, trtr_std = ml_utility(real, train_split_real, model(), n_folds=n_folds, scoring=scoring)
            all_scores[f"utility-{model_name}-{scoring}-trtr"] = {"mean": trtr_mean, "std": trtr_std}


    ### TSTR
    # copy the train and test datasets
    test_split       = real.data_test.copy(deep=True)
    train_split_real = real.data.copy(deep=True)
    fake_data        = fake.copy(deep=True)

    # remove the columns where all the values are nan
    for c in fake_data.columns:
        if fake_data[c].isna().all():
            fake_data = fake_data.drop(columns=[c], axis=1)
            train_split_real = train_split_real.drop(columns=[c], axis=1)
            test_split = test_split.drop(columns=[c], axis=1)

    # remove the duplicates and the number of collisions
    # fake_data, _ = get_df_without_collisions_duplicates(real, fake_data)
    fake_data = get_df_no_collision(real, fake_data)

    # create the train and test datasets with correct number of data
    # downsample the sythetic data to the minimum size
    fake_data = fake_data.sample(n=n_data_limit, random_state=42)

    for model_name, model in classif_models.items():
        for scoring in scorings:
            # check that we have more than one class in the target
            if len(fake[real.target].unique()) <= 1:
                trtr_mean, trtr_std, tstr_mean, tstr_std = 0, 0, 0, 0
            else:
                tstr_mean, tstr_std = ml_utility(real, fake_data, model(), n_folds=n_folds, scoring=scoring)
            all_scores[f"utility-{model_name}-{scoring}-tstr"] = {"mean": tstr_mean, "std": tstr_std}


    ### Combined
    # copy the train and test datasets
    test_split       = real.data_test.copy(deep=True)
    train_split_real = real.data.copy(deep=True)
    fake_data        = fake.copy(deep=True)

    # remove the duplicates from the fake_data (wants to keep the collisions for basic data augmentation)
    # fake_data = fake_data.drop_duplicates()

    # remove the columns where all the values are nan
    for c in fake_data.columns:
        if fake_data[c].isna().all():
            fake_data = fake_data.drop(columns=[c], axis=1)
            train_split_real = train_split_real.drop(columns=[c], axis=1)
            test_split = test_split.drop(columns=[c], axis=1)
    
    # create the train and test datasets with correct number of data
    # downsample the sythetic data to the minimum size
    fake_data = fake_data.sample(n=n_data_limit, random_state=42)

    # combined train df is made of 50% real and 50% fake and size == n_data_limit
    fake_sample = fake_data.sample(n=n_data_limit//2, random_state=42)
    real_sample = train_split_real.sample(n=n_data_limit//2, random_state=42)
    train_df = pd.concat([real_sample, fake_sample], axis=0, ignore_index=True)

    for model_name, model in classif_models.items():
        for scoring in scorings:
            # check that we have more than one class in the target
            if len(fake[real.target].unique()) <= 1:
                mean_score, std_score = 0, 0
            else:
                mean_score, std_score = ml_utility(real, train_df, model(), n_folds=n_folds, scoring=scoring)
            all_scores[f"utility-{model_name}-{scoring}-combined"] = {"mean": mean_score, "std": std_score}


    ############
    ### prdc ###
    ############
    print("running prdc", flush=True)
    limit = min(n_data_limit, 350)
    prdc_scores = prdc(real, fake, nearest_k=nearest_k, n_folds=n_folds, max_data=limit)
    for key, value in prdc_scores.items():
        all_scores[key] = value

    if verbose:
        print_results(all_scores, gen_model_name)

    return all_scores
    

def eval_dataframes(dname, fake_frames, fake_names, filename, test_name):
    # compute limit size of data due to duplicates

    LIMIT_SIZE_FAKE = 1e6
    model_min = None
    real_df = Dataloader(dname, parent_directory="data")
    collision_scores = {}

    for i, fake_frame in enumerate(fake_frames):
        fake = fake_frame.copy(deep=True)
        # remove columns with only nan values
        for c in fake.columns:
            if fake[c].isna().all():
                fake = fake.drop(columns=[c], axis=1)

        tmp_df, n_collisions = get_df_without_collisions_duplicates(real_df, fake.drop_duplicates())
        tmp_df2 = get_df_no_collision(real_df, fake)
        collision_scores[fake_names[i]] = len(fake) - len(tmp_df2)
        actual_size = len(tmp_df2)
        print(f"{fake_names[i]}: {actual_size}  -  size no duplicate, no collision: {len(tmp_df)}")
        if actual_size < LIMIT_SIZE_FAKE:
            LIMIT_SIZE_FAKE = actual_size
            model_min = fake_names[i]
    if len(real_df.data) < LIMIT_SIZE_FAKE:
        LIMIT_SIZE_FAKE = len(real_df.data)
        model_min = "real"
    print("--"*20)
    print(f"LIMIT SIZE: {LIMIT_SIZE_FAKE}, model min: {model_min}", flush=True)
    print("--"*20)

    if LIMIT_SIZE_FAKE == 0:
        raise ValueError("LIMIT_SIZE_FAKE is 0. No data to evaluate")

    all_scores = []
    for model_name, fake in zip(fake_names, fake_frames):
        real = Dataloader(dname, parent_directory="data")
        n_dup = len(fake) - len(fake.drop_duplicates())
        og_fake_size = len(fake)
        no_dup_fake_size = len(fake.drop_duplicates())
        
        # remove the duplicates ??
        # fake = fake.drop_duplicates()

        print(f"--Running {model_name}", flush=True)

        scores = evaluate_synth_data(real, fake, model_name, test_name, n_data_limit=LIMIT_SIZE_FAKE, n_folds=5, test_size=0.2, nearest_k=5, show_plots=False, verbose=False)
        scores["Duplicates"] = n_dup/og_fake_size * 100
        scores["Collisions"] = collision_scores[model_name]/og_fake_size * 100
        scores["Data size"] = LIMIT_SIZE_FAKE
        scores["Limit model"] = model_min
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





##########################################################################################################################################





if __name__ == "__main__":

    ALL_DATASETS = ["adult", "thyroid", "german", "sick", "bank", "travel"]

    ########################################################################
    ####################     COMPETITORS COMPARISON     ####################
    ########################################################################

    DONE = False

    if not DONE:
        test_cat = "competitors_comparison"

        fake_names = ["statgen", "ctgan", "tabddpm", "great", "tabula", "epic", "synthloop", "reduced", "promptrefine"]

        for dname in ALL_DATASETS:
            print("=="*20)
            print(f"RUNNING: {test_cat} on {dname}", flush=True)
            print("=="*20)

            # try:
            fake_frames = []
            for name in fake_names[:-3]:
                fake_frames.append(pd.read_csv(f"generated_examples/{name}_gen_data_{dname}.csv"))
            for name in fake_names[-3:]:
                fake_frames.append(pd.read_csv(f"generated_examples/{name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
            filename = f"evaluation_gen_data/{dname}/competitors_comparison.csv"
            eval_dataframes(dname,
                            fake_frames,
                            fake_names,
                            filename,
                            f"competitors_comparison_{dname}")
            print(f"saved as: {filename}", flush=True)
            # except Exception as e:
            #     print(f"ERROR: {e}", flush=True)


    # eval with 40 data in training
    DONE = False

    if not DONE:
        test_cat = "competitors_comparison_40_training"

        fake_names = ["ctgan", "tabddpm", "synthloop", "reduced", "promptrefine"]

        for dname in ["thyroid"]:
            print("=="*20)
            print(f"RUNNING: {test_cat} on {dname}", flush=True)
            print("=="*20)

            fake_frames = []
            for name in fake_names:
                if name == "ctgan" or name == "tabddpm":
                    fake_frames.append(pd.read_csv(f"generated_examples/{name}_gen_data_{dname}_limit40.csv"))
                else:
                    fake_frames.append(pd.read_csv(f"generated_examples/{name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
            filename = f"evaluation_gen_data/{dname}/competitors_comparison_40_training.csv"
            eval_dataframes(dname,
                            fake_frames,
                            fake_names,
                            filename,
                            f"competitors_comparison_40_training_{dname}")
            print(f"saved as: {filename}", flush=True)




    ################################################################
    ####################     LLM COMPARISON     ####################
    ################################################################

    DONE = False

    if not DONE:
        test_cat = "llm_comparison"

        gen_llms = ["llama3.1", "gpt4o", "deepseek-v3"]
        model_names = ["synthloop", "reduced", "promptrefine"]

        fake_names = ["synthloop_llama3.1", "synthloop_gpt4o", "synthloop_deepseek-v3",
                    "reduced_llama3.1", "reduced_gpt4o", "reduced_deepseek-v3",
                    "promptrefine_llama3.1", "promptrefine_gpt4o", "promptrefine_deepseek-v3"]
        
        # fake_names = ["synthloop_llama3.1", "synthloop_gpt4o-mini",
        #               "reduced_llama3.1", "reduced_gpt4o-mini",
        #               "promptrefine_llama3.1", "promptrefine_gpt4o-mini"]

        for dname in ["adult", "bank", "thyroid"]:
            print("=="*20)
            print(f"RUNNING: {test_cat} on {dname}", flush=True)
            print("=="*20)

            # try:
            fake_frames = []
            for model_name in model_names:
                for gen_llm in gen_llms:
                    fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_{gen_llm}_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
            filename = f"evaluation_gen_data/{dname}/llm_big_comparison.csv"
            eval_dataframes(dname,
                            fake_frames,
                            fake_names,
                            filename,
                            f"llm_big_comparison_{dname}")
            print(f"saved as: {filename}", flush=True)
            # except Exception as e:
            #     print(f"ERROR: {e}", flush=True)


    #################################################################
    ####################     ITER COMPARISON     ####################
    #################################################################

    DONE = False

    if not DONE:
        test_cat = "iter_comparison"

        fake_names = ["iter_0", "iter_1", "iter_2"]
        gen_llms = ["llama3.1", "gpt4o-mini", "deepseek-v3"]

        dataset_round = { "llama3.1": {"adult": 7,
                                    "thyroid": 16,
                                    "german": 6,
                                    "sick": 17,
                                    "bank": 7,
                                    "travel": 2},
                        "gpt4o-mini": {"adult": 105,
                                    "thyroid": 96,
                                    "german": 95,
                                    "sick": 88,
                                    "bank": 110,
                                    "travel": 87},
                        "deepseek-v3": {"adult": 74,
                                    "thyroid": 73,
                                    "german": 59,
                                    "sick": 70,
                                    "bank": 63,
                                    "travel": 70}
                        }

        for gen_llm in gen_llms:
            for dname in ALL_DATASETS:
                print("=="*20)
                print(f"RUNNING: {test_cat} on {dname} ({gen_llm})", flush=True)
                print("=="*20)

                # try:
                folder_name = f"generated_examples/synthloop_all_iters/synthloop_{dname}_baseline"
                if gen_llm == "deepseek-v3" or gen_llm == "gpt4o-mini":
                    folder_name += f"_{gen_llm}"
                template_name = "synthloop_{dname}_{i_round}_iter_{i_iter}_" + gen_llm + "_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotsFbFalse_epicFalse_featoriginal"
                df_iters = merge_dataset_iterations(folder_name,
                                                    dname,
                                                    template_name,
                                                    n_iter=3,
                                                    n_round=dataset_round[gen_llm][dname])
                fake_frames = [df_iters[df_iters["iter"] == i].drop(columns=["iter"], axis=1) for i in range(3)]
                filename = f"evaluation_gen_data/{dname}/iter_comparison_{gen_llm}.csv"
                eval_dataframes(dname,
                                fake_frames,
                                fake_names,
                                filename,
                                f"iter_comparison_{dname}_{gen_llm}")
                print(f"saved as: {filename}", flush=True)
                # except Exception as e:
                #     print(f"ERROR: {e}", flush=True)


    #####################################################################
    ####################     VARIANTS COMPARISON     ####################
    #####################################################################

    DONE = False

    if not DONE:
        test_cat = "variants_comparison"

        fake_names = ["baseline",
                    "temp. 0.9",
                    "30 shots",
                    "cat first",
                    "num first",
                    # "EPIC prompting",
                    "fshots feedback",
                    "sentence"]
        
        model_names = ["synthloop", "reduced", "promptrefine"]
        
        for model_name in model_names:
            for dname in ["adult", "thyroid", "travel"]:
                print("=="*20)
                print(f"RUNNING: {test_cat} on {dname} ({model_name})", flush=True)
                print("=="*20)

                # try:
                fake_frames = []
                # baseline
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                # temp 0.9
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.9_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                # 30 shots
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots30_shotFbFalse_orderoriginal_epicFalse.csv"))
                # cat first
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_ordercat_first_epicFalse.csv"))
                # num first
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_ordernum_first_epicFalse.csv"))
                # EPIC prompting
                # fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicTrue.csv"))
                # fshots feedback
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbTrue_orderoriginal_epicFalse.csv"))
                # sentence
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_sentence_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                filename = f"evaluation_gen_data/{dname}/{model_name}_variants_comparison.csv"
                eval_dataframes(dname,
                                fake_frames,
                                fake_names,
                                filename,
                                f"variants_comparison_{dname}_{model_name}")
                print(f"saved as: {filename}", flush=True)
                # except Exception as e:
                #     print(f"ERROR: {e}", flush=True)


    # evaluate the synthloop and refine together
    DONE = False

    if not DONE:
        test_cat = "variants_comparison"

        fake_names = ["synthloop_baseline", "synthloop_temp. 0.9", "synthloop_30 shots", "synthloop_cat first", "synthloop_num first", "synthloop_fshots feedback", "synthloop_sentence",
                    "promptrefine_baseline", "promptrefine_temp. 0.9", "promptrefine_30 shots", "promptrefine_cat first", "promptrefine_num first", "promptrefine_fshots feedback", "promptrefine_sentence"]
        
        model_names = ["synthloop", "promptrefine"]

        for dname in ["adult", "thyroid", "travel"]:
            print("=="*20)
            print(f"RUNNING: {test_cat} on {dname}", flush=True)
            print("=="*20)

            fake_frames = []
            for model_name in model_names:

                # baseline
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                # temp 0.9
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.9_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                # 30 shots
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots30_shotFbFalse_orderoriginal_epicFalse.csv"))
                # cat first
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_ordercat_first_epicFalse.csv"))
                # num first
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_ordernum_first_epicFalse.csv"))
                # EPIC prompting
                # fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicTrue.csv"))
                # fshots feedback
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbTrue_orderoriginal_epicFalse.csv"))
                # sentence
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_sentence_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
            filename = f"evaluation_gen_data/{dname}/synthloop-refine_variants_comparison.csv"
            eval_dataframes(dname,
                            fake_frames,
                            fake_names,
                            filename,
                            f"variants_comparison_{dname}_synthloop-refine")
            print(f"saved as: {filename}", flush=True)
            # except Exception as e:
            #     print(f"ERROR: {e}", flush=True)



    ##########################################################################
    ####################     INFO-WEAKNESS COMPARISON     ####################
    ##########################################################################

    DONE = False

    if not DONE:
        test_cat = "info-weakness_comparison"

        fake_names = ["info-full", "info-weakness", "noInfo-full", "noInfo-weakness"]

        model_names = ["synthloop", "reduced", "promptrefine"]

        for model_name in model_names:
            for dname in ["adult", "thyroid", "travel"]:
                print("=="*20)
                print(f"RUNNING: {test_cat} on {dname} ({model_name})", flush=True)
                print("=="*20)

                # try:
                fake_frames = []
                # info-full 
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                # info-weakness
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyTrue_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                # noInfo-full
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoTrue_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                # noInfo-weakness
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoTrue_weaknessOnlyTrue_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                filename = f"evaluation_gen_data/{dname}/{model_name}_info-weakness_comparison.csv"
                eval_dataframes(dname,
                                fake_frames,
                                fake_names,
                                filename,
                                f"info-weakness_comparison_{dname}_{model_name}")
                print(f"saved as: {filename}", flush=True)
                # except Exception as e:
                #     print(f"ERROR: {e}", flush=True)


    # evaluate synthloop and refine together
    DONE = False
    if not DONE:

        test_cat = "variants_comparison (synthloop-refine)"
        fake_names = ["synthloop_info-full", "synthloop_info-weakness", "synthloop_noInfo-full", "synthloop_noInfo-weakness",
                      "promptrefine_info-full", "promptrefine_info-weakness", "promptrefine_noInfo-full", "promptrefine_noInfo-weakness"]
        model_names = ["synthloop", "promptrefine"]

        for dname in ["adult", "thyroid"]:
            print("=="*20)
            print(f"RUNNING: {test_cat} on {dname}", flush=True)
            print("=="*20)

            fake_frames = []
            for model_name in model_names:
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoFalse_weaknessOnlyTrue_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoTrue_weaknessOnlyFalse_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
                fake_frames.append(pd.read_csv(f"generated_examples/{model_name}_{dname}_llama3.1_3iters_csv_noInfoTrue_weaknessOnlyTrue_temp0.7_shots20_shotFbFalse_orderoriginal_epicFalse.csv"))
            filename = f"evaluation_gen_data/{dname}/synthloop-refine_info-weakness_comparison.csv"
            eval_dataframes(dname,
                            fake_frames,
                            fake_names,
                            filename,
                            f"variants_comparison_{dname}_synthloop-refine")
            print(f"saved as: {filename}", flush=True)












