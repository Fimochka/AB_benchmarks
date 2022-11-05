import os
import pandas as pd
import numpy as np
import dill
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def calc_synth_metrics(pilot_store=None,
                       prefix='prepilot',
                       period_df=None,
                       target_cols=list(),
                       plants_control_dict=dict(),
                       input_models_path=None):
    """
    For a selected pilot plant loads regressions and calcs values for every metric from target_cols
    ---------------
    Parameters:
    pilot_store: int (default = None)
        Pilot plant name (ex. 123)
    prefix: str (default = prepilot)
        We have 2 sets of regressions:
         1) for a history period to predict a prepilot period
         2) for a history period including a prepilot to predict a pilot period
    period_df: pandas DataFrame (default = None)
        Pandas DataFrame containing data. Should have columns ['plant', 'calday'] + target_cols
        It can be a prepilot_df or pilot_df
    target_cols: python list (default = [])
        Whar metrics we need to make prediction for
    plants_control_dict: python dict (default = {})
        Dictionary containing control plants for every pilot plant
    input_models_path: str (default = None)
        Path to the preptrained regression models
    ----------------
    Returns:
    plant_synth_df: pandas DataFrame
        Contains ['plant', 'calday'] + target_cols, where:
            plant is modified to plant+'0' (synthetic_id)
            every col from target_cols is an array of predicted values using regression
    """

    plant_preds = []
    control_stores = plants_control_dict[pilot_store]
    for metric in target_cols:
        model_path = [path for path in os.listdir(input_models_path) if prefix in path and metric in path]
        plant_names = [str(path.split(prefix)[1].split("_")[1]) for path in model_path]
        pilot_models = dict(zip(plant_names, model_path))

        pilot = period_df[period_df['country_code'] == pilot_store].set_index('dt').sort_index()
        control = period_df[period_df['country_code'].isin(control_stores)].set_index('dt').sort_index()
        pilot_common_indexes = set(pilot.index.values).intersection(set(control.index.values))
        pilot = pilot[pilot.index.isin(pilot_common_indexes)]
        control = control[control.index.isin(pilot_common_indexes)]

        X_test = control.pivot(columns='country_code', values=metric).fillna(0)[control_stores]
        try:
            y_preds = dill.load(open(input_models_path + "/" + pilot_models[pilot_store],
                                 'rb')).predict(X_test)
        except Exception as e:
            print(X_test)
            raise e

        plant_preds.append(y_preds)

    plant_synth_df = pd.DataFrame(plant_preds).T
    plant_synth_df.columns = target_cols
    plant_synth_df['country_code'] = str(pilot_store)# + "0")
    plant_synth_df['dt'] = pilot.index.values
    plant_synth_df = plant_synth_df[['country_code', 'dt'] + target_cols]
    return plant_synth_df

def give_forecast(history_pilot,
                  history_control,
                  prepilot_pilot,
                  prepilot_control,
                  target_col,
                  params,
                  control_stores):
    assert isinstance(history_pilot, pd.DataFrame)
    assert isinstance(history_control, pd.DataFrame)
    assert isinstance(prepilot_pilot, pd.DataFrame)
    assert isinstance(prepilot_control, pd.DataFrame)
    y_train = history_pilot[target_col].replace(np.inf, 0)
    y_test = prepilot_pilot[target_col].replace(np.inf, 0)
    X_train = history_control.pivot(columns='country_code',
                                    values=target_col).replace(np.inf, 0)\
        .astype('float32')\
        .fillna(0)[control_stores]
    X_test = prepilot_control.pivot(columns='country_code',
                                    values=target_col).replace(np.inf, 0)\
        .astype('float32')\
        .fillna(0)[control_stores]

    model = ElasticNet(**params)
    try:
        model.fit(X_train.astype('float16').values,
                  y_train)
    except Exception as e:
        print(X_train.shape, X_train.dtypes)
        print(y_train.replace(np.inf, 0))
        print(history_control['country_code'].unique())
        raise e
    try:
        predict = model.predict(X_test)
    except Exception as e:
        print(X_train)
        print(history_control['country_code'].unique())
        raise e
    err_mae = mean_absolute_error(y_test, predict)
    err_mse = mean_squared_error(y_test, predict)
    err_r2 = r2_score(y_test, predict)
    return {'predict': predict,
            'MAE': err_mae,
            'truth': y_test,
            'MSE': err_mse,
            'R2': err_r2,
            'model': model}
