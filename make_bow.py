"""Classify accruals.


TODO: Write models to disk. Skip classification if they already exist.
TODO: Consider not making list of BoW but do one at a time, run all algos, then start again.
TODO: look up how to do scoring right – what to score, all, test or train?
TODO: write csv report of results

"""

import csv
import os
import pickle
import re
import sys

import numpy as np
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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


def make_vectorizers():
    """Initialize the parameters of various vectorizers, return list of these."""
    """Make various vectorizers based on parameter permutations, return a
    list of them.

    `testing` parameter only returns one vectorizer.
    """
    vectorizers_list = []
    vectorizer_opts = [
                       # {'analyzer_opt': 'char_wb',
                       #  'lowercase_opts': [True, False],
                       #  'ngram_range_opts': [(3, 6), (3, 9), (3, 12)]},
                       {'analyzer_opt': 'word',
                        'lowercase_opts': [True], # , False],
                        'ngram_range_opts': [(1, 1), (1, 2), (1, 3)]}]
    for vec_opt in vectorizer_opts:
        analyzer = vec_opt['analyzer_opt']
        for case_opt in vec_opt['lowercase_opts']:
            for ngram_opt in vec_opt['ngram_range_opts']:
                # vectorizer = CountVectorizer(analyzer=analyzer,
                #                              ngram_range=ngram_opt,
                #                              lowercase=case_opt)
                # vectorizers_list.append(vectorizer)
                vectorizer = TfidfVectorizer(analyzer=analyzer,
                                             ngram_range=ngram_opt,
                                             lowercase=case_opt)
                vectorizers_list.append(vectorizer)
    return vectorizers_list


def get_target_column(df, target_column):
    """Return just the column of label values."""
    return df[target_column]


def make_bow(df, text_column, vectorizer, overwrite=None):
    """Take text column, check for previous BoW, if there skip; if not make and save"""
    params_dict = vectorizer.get_params()
    analyzer = params_dict['analyzer']
    lowercase = params_dict['lowercase']
    ngram_range = params_dict['ngram_range']
    ngram_1 = ngram_range[0]
    ngram_2 = ngram_range[1]
    bow_name = 'analyzer_{0}|lowercase_{1}|ngrams_{2}-{3}.df'.format(analyzer,
                                                                     lowercase,
                                                                     ngram_1,
                                                                     ngram_2)
    fp_bow = os.path.join(dir_models, bow_name)

    # check if it's in there
    if not os.path.isdir(dir_models):
        os.mkdir(dir_models)
    models_dfs = os.listdir(dir_models)

    # if it's there, load and return
    if bow_name in models_dfs and not overwrite:
        print('Loading BoW from: "{}"'.format(fp_bow))
        with open(fp_bow, 'rb') as fo:
            X = pickle.load(fo)
        print('Shape:', X.shape)

    print('Making BoW and saving at: "{}"'.format(fp_bow))

    list_text_column = df[text_column].tolist()

    document_term_matrix = vectorizer.fit_transform(list_text_column)  # scipy.sparse.csr.csr_matrix

    # Convert scipy.sparse.csr.csr_matrix to Dataframe,
    # since Pandas does not accept this Scipy type
    # http://stackoverflow.com/a/17819427

    X = pandas.SparseDataFrame([pandas.SparseSeries(document_term_matrix[i].toarray().ravel())
                                for i in np.arange(document_term_matrix.shape[0])])

    # write
    with open(fp_bow, 'wb') as fo:
        pickle.dump(X, fo)
    print('Shape:', X.shape)


def make_save_all_bow():
    """Do everything, from reading file to saving final BoW."""
    # all 2016 accrual re-classifications, I think
    csv_path = '../marritt_data/20170222_accruals/all-filtered_cleaned.csv'
    df_marriott = csv_to_df(csv_path)
    list_of_vectorizers = make_vectorizers()

    # get previously generated scores
    all_scores_csv_fp = os.path.join(dir_models, 'previous_scores.csv')
    previous_scores = {}
    if os.path.isfile(all_scores_csv_fp):
        with open(all_scores_csv_fp) as fo:
            csv_reader = csv.reader(fo)
        for row in csv_reader:
            model_name = row[0]
            bow_name = row[1]
            prev_score = row[2]
            previous_scores[model_name] = (bow_name, prev_score)

    # make classifiers and score
    for vectorizer in list_of_vectorizers:
        make_bow(df=df_marriott, text_column='LineDescr', vectorizer=vectorizer)


def rm_months_func(x):
    """Take str, rm month or sim."""
    # \bNov.+?\b
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months_str = '\\b' + '.+?\\b|\\b'.join(months) + '.+?\\b'
    comp = re.compile(months_str, re.IGNORECASE)
    return comp.sub('', x)


def rm_yrs_func(x):
    comp = re.compile(r'\b[0-9]{4}\b', re.IGNORECASE)
    return comp.sub('', x)


def rm_numbers_func(x):
    comp = re.compile(r'\b[0-9]\b', re.IGNORECASE)
    return comp.sub('', x)


def make_baseline_feature_table(csv_fp, df_fp, text_column='LineDescr', overwrite=False, rm_months=False, rm_yrs=False, rm_numbers=False):
    """Simple version to test against."""

    df_marriott = csv_to_df(csv_fp)

    vectorizer = TfidfVectorizer(analyzer='word',
                                 ngram_range=(1,3),
                                 lowercase=True)
    baseline_df_fp = os.path.join(dir_models, df_fp)
    # make_bow(df=df_marriott, text_column=text_column, vectorizer=vectorizer)
    list_text_column = df_marriott[text_column].tolist()

    if rm_months:
        list_text_column = [rm_months_func(_str) for _str in list_text_column]

    if rm_yrs:
        list_text_column = [rm_yrs_func(_str) for _str in list_text_column]

    if rm_numbers:
        list_text_column = [rm_numbers_func(_str) for _str in list_text_column]

    document_term_matrix = vectorizer.fit_transform(list_text_column)  # scipy.sparse.csr.csr_matrix

    # Convert scipy.sparse.csr.csr_matrix to Dataframe,
    # since Pandas does not accept this Scipy type
    # http://stackoverflow.com/a/17819427
    X = pandas.SparseDataFrame([pandas.SparseSeries(document_term_matrix[i].toarray().ravel())
                                for i in np.arange(document_term_matrix.shape[0])])

    if os.path.isfile(baseline_df_fp) and not overwrite:
        print('Refusing to overwrite existing df file.')
        return
    else:
        with open(baseline_df_fp, 'wb') as fo:
            pickle.dump(X, fo)


def make_extended_ap_spreadsheet(accruals_csv, ap_csv, extended_csv):
    """Match similar rows from two dfs, then concat the NL descriptions, then save the
    df as csv. After this function, use the BoW vectorizer as usual.
    """
    df_accruals = csv_to_df(accruals_csv)
    # print(df_accruals.shape)  # (2445, 10)
    len_rows_accruals = df_accruals.shape[0]  # 2445
    df_accruals['accurals_index_col'] = df_accruals.index
    df_ap = df = pandas.read_excel(ap_csv, sheetname='sheet1')
    # print(df_ap.shape)  # (36259, 17)

    df_merged = pandas.merge(df_accruals, df_ap,
                           left_on=['OperUnit', 'DeptID', 'Account', 'Amount'],
                           right_on=['Oper Unit', 'Dept', 'Account', 'Amount'])
    # print(df_merged.shape)  # (748, 25)
    #df_merged.to_csv('merged_test.csv')

    #? now reduce further by the 3 date fields in df_ap?
    # 'Journal Date', 'Voucher Creation Date', 'Invoice Date' (df_ap); 'Date' (df_accruals)
    #! Date has floats '42735.0'

    series_concat = df_merged["LineDescr"].map(str) + ' ' + df_merged["Name"].map(str) + ' ' + df_merged["Descr.1"].map(str)

    # finally write to file
    df_merged_plus_desc = df_merged.assign(NewLineDescr=series_concat.tolist())
    print('Columns of new DataFrame being written to "{0}": {1}'.format(extended_csv, df_merged_plus_desc.columns.values))

    df_merged_plus_desc.to_csv(extended_csv)
