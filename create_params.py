import pandas as pd

params_dict = {
    "res_attn":{
    "device": "cuda",
    "batch_size": 16,
    "num_workers": 4,
    },
    "capsnet":{
    "device": "cuda",
    "batch_size": 16,
    "num_workers": 4,   
    }
}

pd.DataFrame.from_dict(params_dict, orient='index').to_csv('params.csv')
