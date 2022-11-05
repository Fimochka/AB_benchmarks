import pandas as pd
import numpy as np

from src.data.exceptions import UnknownLinearisationException


def calc_linearisation_mean(history_df: pd.DataFrame = None,
                                 current_df: pd.DataFrame = None,
                                 metrics: list = None,
                                 linearisation_dictionary: dict = None
                                 ):
    """
    Does linearisation for input metrics using dayofweek aggregations based on the previous year
    Parameters:
    --------------
    history_df: pandas DataFrame (default = None)
        Contains data for a previous year (we will calc stats on it)
    current_df: pandas DataFrame (default = None)
        Contains data for a current period (we will do linearisation on it)
    metrics: list (default = None)
        Python list of columns for which we do a linearisation
    linearisation_dictionary: dict (default = None)
        Python dictionary containing linear_type for every column
    Returns:
    ---------------
    current_df: pandas DataFrame
        Dataframe containing modified metrics
    """
    for target_col in linearisation_dictionary:
        linear_type = linearisation_dictionary[target_col]
        if linear_type is None:
            current_df[target_col] = current_df[target_col].fillna(method='ffill').dropna()
        elif linear_type == 'global_mean':
            current_df[target_col] = current_df[target_col].fillna(method='ffill').dropna()
            current_df[target_col] = current_df[target_col]-current_df[target_col].mean()
        elif linear_type == 'dayofweek':
            stores_dayofweek_mean = history_df.groupby(['plant',
                                                        'dayofweek'])\
                .agg({target_col: lambda x: np.mean(x)})\
                .reset_index()\
                .rename(columns={target_col: '{}_mean'.format(target_col)})

            current_df = pd.merge(current_df.reset_index(),
                                  stores_dayofweek_mean,
                                  on=['plant',
                                      'dayofweek'])

            current_df[target_col] = current_df[target_col].fillna(method='ffill').dropna()
            current_df.loc[:, target_col] = current_df.loc[:,
                                            target_col] - current_df.loc[:,
                                                          '{}_mean'.format(target_col)]
            current_df = current_df.drop(['index'], 1)
        elif linear_type == 'mean_plant':
            stores_mean_plant = history_df.groupby(['plant']) \
                .agg({target_col: lambda x: np.mean(x)}) \
                .reset_index() \
                .rename(columns={target_col: '{}_mean'.format(target_col)})
            current_df = pd.merge(current_df.reset_index(),
                                  stores_mean_plant,
                                  on=['plant'])
            current_df[target_col] = current_df[target_col].fillna(method='ffill').dropna()
            current_df.loc[:, target_col] = current_df.loc[:,
                                            target_col] - current_df.loc[:,
                                                          '{}_mean'.format(target_col)]
            current_df = current_df.drop(['index'], 1)
        elif linear_type == 'rolling':
            current_df[target_col] = current_df[target_col].fillna(method='ffill')\
                .rolling(7,
                         2)\
                .mean()\
                .dropna()
        else:
            raise UnknownLinearisationException("Unknown type of linearisation!"
                                                " '{}' is not supported. Possible variants: "
                                                "None/mean_plant/dayofweek/rolling!".format(linear_type))
    return current_df
