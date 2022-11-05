def sum_aggregate_selected_metric(pilot=None,
                         prepilot=None,
                         control=None,
                         precontrol=None,
                         metric=None):
    """
    Aggregates data by ['plant', 'calday'] and calcs sum for the passed metric
    ------------
    Parameters:
    pilot: pandas DataFrame (default = None)
        experiment plants dataset for a test (pilot) period
    prepilot: pandas DataFrame (default = None)
        experiment plants dataset for a prepilot period
    control: pandas DataFrame (default = None)
        control plants dataset for a test (pilot) period
    precontrol: pandas DataFrame (default = None)
        control plants dataset for a prepilot period
    metric: str (default = None)
        metric we need to aggregate
    ------------
    Returns:
    agg_data: python list
        contains 4 datasets:
            pilot_ (experiment plants stats for a pilot period)
            prepilot_ (experiment plants stats for a PREpilot period)
            control_ (control plants stats for a pilot period)
            precontrol_ (control plants stats for a PREpilot period)
    """
    pilot_ = pilot.groupby(['country_code',
                            'dt'])[metric].sum().reset_index(name=metric)
    prepilot_ = prepilot.groupby(['country_code',
                                  'dt'])[metric].sum().reset_index(name=metric)

    control_ = control.groupby(['country_code',
                                'dt'])[metric].sum().reset_index(name=metric)
    precontrol_ = precontrol.groupby(['country_code',
                                      'dt'])[metric].sum().reset_index(name=metric)

    agg_data = pilot_, control_, prepilot_, precontrol_
    return agg_data
