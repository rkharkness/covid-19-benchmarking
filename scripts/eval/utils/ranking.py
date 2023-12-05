import json 
import glob 
import argparse 
import numpy as np

def read_json(path): 
    with open(path) as f:
        return json.load(f)

def ranking(results_dicts, metric):
    metric_dict = {}
    for name in results_dicts.keys():
#          print(name)
#          print(results_dicts[name]['All'])
   #       print(results_dicts[name]['All'][metric])
          metric_dict[name] = np.mean(results_dicts[name]['All'][metric])

 #   print('sorted')
#    print(metric_dict['ssl_am']) #, reverse=True, key=lambda item: item[1]))

    ranking = {k: v for k, v in sorted(metric_dict.items(), reverse=False, key=lambda item: item[1])}
    #ranking = sorted(metric_dict, key=metric_dict.get, reverse=True)
  #  print(ranking,'r')
    ranking_dict = {k:v for k,v in enumerate(ranking)}
    ranking_dict = {v:k for k,v in ranking_dict.items()}
    print(ranking_dict)
    return ranking_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script') 
    parser.add_argument('--subpop', default=False, type=bool, help='One of: Ethnicity, Comorbidity, Smoker, Gender, Age')
    parser.add_argument('--test', default='ltht', type=str, help='One of: ltht, covid_qu_ex, nccid_test, ltht_no_pneumonia')
    args = parser.parse_args()

    test_map = {'nccid_test':'NCCID Test', 'ltht':'LTHT', 'ltht_no_pneumonia': 'LTHT (NP)', 'ltht_pneumonia':'LTHT (P)', 'covidgr':'COVIDGR'}
    
    results_paths = glob.glob(f"/MULTIX/DATA/HOME/covid-19-benchmarking/results/*/{args.test}/*All.json")
    model_names = [r.split('/')[6] for r in results_paths]
    model_names = [m for m in model_names if m not in ['coronet_tfl_seg','xvitcos_seg','ssl_am_seg']]
    results_paths = [m for m in results_paths if 'seg' not in m]

    results_dicts = {}
    for name, r in zip(model_names,results_paths):
        print(name, r)
        results_dicts[name] = read_json(r)
    
    metrics = ['acc','recall','precision', 'f1', 'ppv', 'roc_auc']
#    metrics = ['ppv','roc_auc']
    overall_r_dict = {k:[] for k in model_names}
    for m in metrics:

        print(m)
        r_dict = ranking(results_dicts, m)
        for k in r_dict.keys():
            overall_r_dict[k].append(r_dict[k])

    print(overall_r_dict)
    overall_r_dict = {k:np.sum(v) for k,v in overall_r_dict.items()}
    ranking = sorted(overall_r_dict, key=overall_r_dict.get, reverse=True)

    summ_ranking_dict = {k+1:v for k,v in enumerate(ranking)}
    print('OVERALL')
    print(summ_ranking_dict)
    
