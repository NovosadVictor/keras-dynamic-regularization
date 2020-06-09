import os
import pandas as pd
import json


histories_dir = 'histories'

dropout_types = {
    'True': 'trainable',
    'False': 'static',
    'None': 'without_dropout',
}


def make_csv_name(name: str) -> str:
    splitted_name = name.split('_')
    dropout_value = splitted_name[4]
    dropout_type = dropout_types[splitted_name[3]]

    return '{}: {}'.format(dropout_type, dropout_value)


history_files = [name for name in os.listdir(histories_dir) if 'history' in name]
history_files.sort(key=lambda item: (item.split('_')[3], item.split('_')[4]))
parameter_files = [name for name in os.listdir(histories_dir) if 'parameters' in name]
parameter_files.sort(key=lambda item: (item.split('_')[3], item.split('_')[4]))

histories = []
for name in history_files:
    with open(os.path.join(histories_dir, name), 'r') as f:
        histories.append(json.load(f))

parameters = []
for name in parameter_files:
    with open(os.path.join(histories_dir, name), 'r') as f:
        params = json.load(f)
        if params:
            params = [float(name.split('_')[4])] + params
        parameters.append(params)

pd_histories = [
    pd.DataFrame.from_dict(
        {
         **{'{}_{}'.format(make_csv_name(history_files[i]), key): value for key, value in item.items()},
         **{'{}_parameters'.format(make_csv_name(parameter_files[i])): parameters[i]},
        },
        orient='index',
    ).transpose()
    for i, item in enumerate(histories)
]


for i in range(len(pd_histories)):
    pd_histories[i].to_csv(os.path.join(histories_dir, 'csvs', history_files[i][:-4] + 'csv'))
