import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.errors import EmptyDataError
from sklearn.exceptions import NotFittedError


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Parameters:
    ------------
    key : str
        column name
    """

    def __init__(self,
                 key: str = None):
        self.key = key

    def fit(self,
            X: pd.DataFrame,
            y: str = None):
        """
        Does nothing. This method is needed for a compatibility with other sklearn transformers
        """
        return self

    def transform(self,
                  X: pd.DataFrame = None):
        """
        Returns a self.key column from an input Dataframe
        Parameters:
        ------------
        X : pd.DataFrame
            pandas Dataframe from which we need to get a column
        """
        if len(X)==0:
            raise EmptyDataError("Input dataset is empty!")
        try:
            return X[self.key]
        except KeyError as e:
            raise KeyError("Input column {} does not exist in the input Dataset!".format(self.key))

class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    Parameters:
    ------------
    key : str
        column name
    """

    def __init__(self,
                 key: str = None):
        self.key = key

    def fit(self,
            X: pd.DataFrame = None,
            y: str = None):
        """
        Does nothing. This method is needed for a compatibility with other sklearn transformers
        """
        return self

    def transform(self,
                  X: pd.DataFrame = None):
        """
        Returns a self.key column from an input Dataframe
        Parameters:
        ------------
        X : pd.DataFrame
                pandas Dataframe from which we need to get a column
        """
        if len(X)==0:
            raise EmptyDataError("Input dataset is empty!")
        try:
            return X[[self.key]]
        except KeyError as e:
            raise KeyError("Input column {} does not exist in the input Dataset!".format(self.key))

class OHEEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the dataframe to perform OHE on
    Use on discrete variables
    Parameters:
    ------------
    key : str
        column name
    """
    def __init__(self,
                 key: str = None):
        self.key = key
        self.columns = []

    def fit(self,
            X: pd.DataFrame = None,
            y: str = None):
        """
        Uses a pd.get_dummies method to do a OHE on self.key column
        Also saves a self.columns variable (list of final columns AFTER the OHE)
        Parameters:
        ------------
        X : pd.DataFrame (default = None)
            pandas dataframe input data (train) that should contain a self.key column
        y : str (default = None)
            target column name (needed only for a compatibility with other sklearn transformers!)
        """
        if len(X)==0:
            raise EmptyDataError("Input dataset is empty!")
        try:
            self.columns = [col for col in pd.get_dummies(X[[self.key]], prefix=self.key).columns]
        except KeyError as e:
            raise KeyError("Input column {} does not exist in the input Dataset!".format(self.key))
        return self

    def transform(self,
                  X: pd.DataFrame = None):
        """
        Performs a OHE on new data
        Parameters:
        ------------
        X : pd.DataFrame (default = None)
            pandas dataframe input data (test) that should contain a self.key column
        """
        if len(X)==0:
            raise EmptyDataError("Input dataset is empty!")
        if len(self.columns) == 0:
            raise NotFittedError("OHE transformer is not fiited! You need to execute a 'fit' method first!")
        try:
            X = pd.get_dummies(X[[self.key]], prefix=self.key)
            test_columns = [col for col in X.columns]
            for col_ in self.columns:
                if col_ not in test_columns:
                    X[col_] = 0
            return X[self.columns]
        except KeyError as e:
            raise KeyError("Input column {} does not exist in the input Dataset!".format(self.key))


def add_date_attributes(df=None,
                        date_column='dt'):
    """
    Adds to the input DataFrame extra columns: week and dayofweek
    Input DataFrame should contain date_column
    Parameters:
    ----------------
    df: pandas.DataFrame (default = None)
        Input data
    date_column: str (default = calday)
        Name of the date column
    Returns:
    ----------------
    df: pandas.DataFrame
        DataFrame with extra date columns (week and dayofweek)
    """
    df['week'] = df[date_column].dt.isocalendar().week
    df['dayofweek'] = df[date_column].dt.dayofweek
    return df

