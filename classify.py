"""Classification functions"""

import datetime
import os
import pickle
import sys

import numpy as np
import pandas
from sklearn import preprocessing
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from chart_of_accounts_translate import get_chart_of_accounts
from chart_of_accounts_translate import convert_chart_name_to_id

dir_root = os.path.expanduser('~')
dir_models = os.path.join(dir_root, 'marriott_accruals_classification_models')


def csv_to_df(csv_fp):
    """Load csv as dataframe with pandas. Make sure that
    data types are getting loaded correctly. Can add
    column names & types to dtype.
    """
    df = pandas.read_csv(csv_fp, dtype={'Account': np.float64})
    df = df[df['Account'].notnull()]
    return df


def get_target_column(df, target_column):
    return df[target_column]


def make_scaler_from_training_features(training_features):
    """Take training features as df, return scaling model."""
    return preprocessing.StandardScaler().fit(training_features)


def scale_dfs(premade_scaler, features):
    """Take scaler, scale df."""
    return premade_scaler.transform(features)


def do_scaling(X_train, X_test):
    scaler = make_scaler_from_training_features(X_train)
    X_train_scaled = scale_dfs(scaler, X_train)
    X_test_scaled = scale_dfs(scaler, X_test)
    return X_train_scaled, X_test_scaled


def setup_df(bow_name):
    # all 2016 accrual re-classifications, I think
    csv_path = '../marritt_data/20170222_accruals/all-filtered_cleaned.csv'
    df_marriott = csv_to_df(csv_path)
    Y = get_target_column(df_marriott, 'Account')

    # make classifiers and score
    all_scores = {}
    # fp_bow, X = make_bow(df=df_marriott, text_column='LineDescr', vectorizer=vectorizer)


    fp_bow = os.path.join(dir_models, bow_name)
    print('Loading BoW from: "{}"'.format(fp_bow))
    with open(fp_bow, 'rb') as fo:
        X = pickle.load(fo)
    assert X.shape[0] == Y.shape[0], 'Original dataframe and pre-made BoW do not match'

    # train_test_split `test_size` defaults to `0.25`
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    X_train_scaled, X_test_scaled = do_scaling(X_train, X_test)

    return X_train_scaled, X_test_scaled, Y_train, Y_test


def classify(classifier, X_train_scaled, Y_train):
    """Do classifications on variety of algorithms.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    n_estimators = 30
    classifiers_dict = {'decision_tree': DecisionTreeClassifier(),
                        'logistic_classifier': LogisticRegression(), }
    # ! reenamble for testing
    #     classifiers_dict = {'decision_tree': DecisionTreeClassifier(),
    #                         'linear_svc': LinearSVC(),
    #                         'random_forest': RandomForestClassifier(n_estimators=n_estimators),
    #                         'logistic_classifier': LogisticRegression(),
    #                         'gaussian_nb': GaussianNB(),
    #                         # 'multinomial_nb': MultinomialNB(),  # ValueError: Input X must be non-negative
    #                         'bernoulli_nb': BernoulliNB()}
    assert classifier in classifiers_dict.keys(), 'Choose classifier in: {}'.format(classifiers_dict.keys())

    clf = classifiers_dict[classifier]
    clf.fit(X_train_scaled, Y_train)

    return clf


def is_pos(x):
    if x <= 0:
        return True
    else:
        return False


def account_to_coa(account):
    coa_label = get_chart_of_accounts(account)
    return convert_chart_name_to_id(coa_label)


def just_dec_tree(csv_path, df_fp, target_col, extra_fields=None, mk_coa=None, coa_model_fp=None):
    """Run decision tree and return score.

    :param csv_path: The original CSV is needed in order to get the targets, which are not included in the df (tho could be)
    :param df_fp: The actual feature table is here.
    """
    extra_fields = []

    # Load data
    fp_bow = os.path.join(dir_models, df_fp)
    print('Loading BoW from: "{}"'.format(fp_bow))
    with open(fp_bow, 'rb') as fo:
        X = pickle.load(fo)

    df_marriott = csv_to_df(csv_path)  # can del after this
    Y = get_target_column(df_marriott, target_col)

    if mk_coa:
        Y = Y.apply(account_to_coa)

    if 'sign' in extra_fields:
        amounts = get_target_column(df_marriott, 'Amount')
        amounts_polarity = amounts.apply(is_pos)
        X = X.assign(e=amounts_polarity.values)  # http://stackoverflow.com/a/12555510

    if 'amount' in extra_fields:
        amounts = get_target_column(df_marriott, 'Amount')
        # amounts_polarity = amounts.apply(is_pos)
        X = X.assign(e=amounts.values)  # http://stackoverflow.com/a/12555510

    # split and prep data
    # train_test_split `test_size` defaults to `0.25`
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    X_train_scaled, X_test_scaled = do_scaling(X_train, X_test)


    if coa_model_fp:
        coa_model_fp = os.path.join(dir_models, coa_model_fp)
        with open(coa_model_fp, 'rb') as fo:
            coa_clf = pickle.load(fo)
        coa_predicts_train = coa_clf.predict(X_train_scaled)  # <class 'numpy.ndarray'>
        coa_predicts_train = pandas.Series(coa_predicts_train)  # <class 'pandas.core.series.Series'>
        X_train_scaled = pandas.DataFrame(X_train_scaled)  # <class 'pandas.core.series.Series'>
        X_train_scaled = X_train_scaled.assign(e=coa_predicts_train)

        coa_predicts_test = coa_clf.predict(X_test_scaled)  # <class 'numpy.ndarray'>
        coa_predicts_test = pandas.Series(coa_predicts_test)  # <class 'pandas.core.series.Series'>
        X_test_scaled = pandas.DataFrame(X_test_scaled)  # <class 'pandas.core.series.Series'>
        X_test_scaled = X_test_scaled.assign(e=coa_predicts_test)



    # train
    clf = DecisionTreeClassifier()
    clf.fit(X_train_scaled, Y_train)

    # save model
    model_fp = os.path.join(dir_models, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.model')
    print('Saving model at: {}'.format(model_fp))
    with open(model_fp, 'wb') as fo:
        pickle.dump(clf, fo)

    # return score
    score = clf.score(X_test_scaled, Y_test)
    return score
