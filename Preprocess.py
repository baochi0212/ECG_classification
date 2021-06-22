import wfdb
import neurokit2 as nk
import json
import numpy as np
import pandas as pd
import os 
from glob import glob

dx_dict = {
        '426783006' : 'SNR', # Normal sinus rhythm
        '164889003': 'AF', # Atrial fibrillation
        '270492004': 'IAVB', # First-degree atrioventricular block
        '164909002': 'LBBB', # Left bundle branch block
        '713427006': 'RBBB', # Complete right bundle branch block
        '59118001': 'RBBB', # Right bundle branch block
        '284470004': 'PAC', # Premature atrial contraction
        '63593006': 'PAC', # Supraventricular premature beats
        '164884008': 'PVC', # Ventricular ectopics
        '429622005': 'STD', # ST-segment depression
        '164931005': 'STE', # ST-segment elevation
    }


classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']


def gen_data(train_dir):
    if not os.path.exists(os.path.join('formatted/thnig', 'formatted_files')):
        result = []
        _p = []
        _a = []
        _l = []
        _s = []
        for path in (sorted(glob(os.path.join(train_dir, '*.hea')))):
            patient_id = path.split('/')[-1][:-4]
            _, meta_data = wfdb.rdsamp(path[:-4])
            sample_rate = meta_data['fs']
            signal_len = meta_data['sig_len']
            age = meta_data['comments'][0]
            sex = meta_data['comments'][1]
            dx = meta_data['comments'][2]
            age = age[5:] if age.startswith('Age: ') else np.NaN
            sex = sex[5:] if sex.startswith('Sex: ') else 'Unknown'
            dx = dx[4:] if dx.startswith('Dx: ') else ''
            dxs = [dx_dict.get(code) for code in dx.split(',')]
            labels = [0]*9
            for idx, label in enumerate(classes): 
                if label in dxs:
                    labels[idx] = 1
            _p.append(patient_id)
            _a.append(age)
            _l.append(labels)
            _s.append(signal_len)
        for i in [_p, _a, _l, _s]:
            result.append(i)
        dict = {
            'patient_id': result[0],
            'age': result[1],
            'labels' : result[2],
            'signal_len' : result[3]
        }
        with open('formatted/thnig/formatted_files', 'w') as f:
            json.dump(dict, f, indent = 4)



gen_data('data/training/Training_WFDB')


        

