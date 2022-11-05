import numpy as np
import pandas as pd

from src.data.preprocessing.feature_engineering import add_date_attributes


def remove_outliers(input_df=None,
                    history_df=None,
                    low_cutoff=0.5,
                    high_cutoff=2,
                    target_list=list(),
                    verbosity=1,
                    period='history'):
    """
    Calcs outliers based on a prehistoric period
    Parameters:
    ----------------
    input_df: pandas.DataFrame (default = None)
        Dataframe for which we need to remove outliers
    history_df: pandas.DataFrame (default = None)
        'Prehistoric' data (1 year before history_df)
    low_cutoff: float (default = 0.5)
        Low threshold to define outliers
    high_cutoff: float (default = 2)
        High threshold to define outliers
    target_list: python list (default = list())
        Names of target cols we need to find outliers
    verbosity: int (default = 1)
        Will print additional info to stdout if equals to 1
    period: str (default = history)
        Name of a period we filter outliers in
    Returns:
    -----------------
    input_df_filtered: pandas.DataFrame
        Input data without outliers
    outliers_df: pandas.DataFrame
        DataFrame containing all days-"outliers"
    """

    assert isinstance(input_df, pd.DataFrame)
    assert isinstance(history_df, pd.DataFrame)

    input_df = add_date_attributes(input_df)
    history_df = add_date_attributes(history_df)

    #convert float64 to float32
    #float64_cols = input_df.select_dtypes(include=[np.float64]).columns

    #input_df[float64_cols] = input_df[float64_cols].astype('float16')
    #history_df[float64_cols] = history_df[float64_cols].astype('float16')

    prod = pd.merge(input_df,
                    history_df,
                    on=['country_code', 'week', 'dayofweek'],
                    how='left').set_index(['country_code', 'week', 'dayofweek'])

    for target in target_list:
        prod['{}_delta'.format(target)] = prod['{}_x'.format(target)] / prod['{}_y'.format(target)]

    prod['total_min_th'] = prod[['{}_delta'.format(target) for target in target_list]].min(axis=1)
    prod['total_max_th'] = prod[['{}_delta'.format(target) for target in target_list]].max(axis=1)

    outliers = prod.query('total_min_th <= @low_cutoff | total_max_th >= @high_cutoff')
    prod = prod[~prod.index.isin(outliers.index.values)]
    t = input_df.set_index(['country_code', 'week', 'dayofweek'])

    input_df_filtered = t[t.index.isin(prod.index)].reset_index()
    outliers_df = t[t.index.isin(outliers.index)].reset_index()

    outliers_df = pd.merge(outliers_df,
                           outliers[['{}_delta'.format(target) for target in target_list]].reset_index(),
                           on=['country_code', 'week', 'dayofweek'])
    if verbosity:
        print("There are {} ({}%) outliers in {} period".format(len(input_df) - len(input_df_filtered),
                                                                float(len(input_df) - len(input_df_filtered)) / len(
                                                                    input_df),
                                                                period))
    return input_df_filtered, outliers_df