"""Read chart of accounts and re-map the accounts into its basic categories
according to a chart of accounts.
"""

from collections import OrderedDict



from make_bow import csv_to_df

chart_of_accounts = {'Assets': (100000, 199999),
                     'Liabilities': (200000, 284999),
                     'Equity': (285000, 299999),
                     'Sales': (300000, 399999),
                     'Cost of Sales': (400000, 499999),
                     'Wages and Benefits': (500000, 599999),
                     'Other Operating Expenses': (600000, 699999),
                     'Investment Factors and Other Costs': (700000, 799999),
                     'Non - operating Income and Expenses': (800000, 899999),
                     'Statistical Information': (910000, 999999),
                     }


def get_chart_of_accounts(account_id):
    """Take account id, return what bucket in belongs to within Marriott's
    chart of accounts
    """
    for account_group_name, account_group_range in chart_of_accounts.items():
        if account_group_range[0] <= account_id <= account_group_range[1]:
            return account_group_name
        else:
            continue


def convert_chart_name_to_id(chart_name):
    """Take a name of a chart of account group, return a unique id."""
    ordered_coa = OrderedDict(chart_of_accounts)
    r = list(ordered_coa.keys()).index(chart_name)
    return r


def convert_chart_id_to_name(coa_id):
    """Take coa id, return name."""
    ordered_coa = OrderedDict(chart_of_accounts)
    str_name = list(ordered_coa.items())[coa_id]
    return str_name[0]


if __name__ == '__main__':
    from classify import get_target_column
    csv_path = '../marritt_data/20170222_accruals/all-filtered_cleaned.csv'
    df_marriott = csv_to_df(csv_path)
    y = get_target_column(df_marriott, 'Account')
    y = y.astype(int)

    # get stats on Account # distribution
    # print(type(y))  # <class 'pandas.core.series.Series'>
    print('Total instances:', len(y))
    print('Unique account #s:', len(set(y)))
    print('Frequency distribution:')
    # print(y.value_counts())

    for account in y:
        print('Account:', account)
        coa_label = get_chart_of_accounts(account)
        print('CoA label:', coa_label)
        coa_id = convert_chart_name_to_id(coa_label)
        print('CoA id:', coa_id)
        remade_str_name = convert_chart_id_to_name(coa_id)
        print('got name from is?', remade_str_name)
        assert coa_label == remade_str_name, 'Name-id converts broken'
        print('')
