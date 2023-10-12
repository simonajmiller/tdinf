import numpy as np
import json
import sys
sys.path.append('../../../time_domain_gw_inference/')
from utils.io import load_raw_data, get_pe

def get_param_dict(sample): 
    param_names = sample.dtype.names
    p_dict = {k:sample[k] for k in param_names}
    p_dict['f_ref'] = 11
    return p_dict

# Load in the data
ifos = ['H1', 'L1', 'V1']
data_path = '../GW190521_data/{}-{}_GWOSC_16KHZ_R2-1242442952-32.hdf5'
raw_time_dict, raw_data_dict = load_raw_data(ifos=ifos,path=data_path)
_, pe_samples, log_prob, _, _  = get_pe(raw_time_dict, verbose=False)

# Locate and save maximum posterior sample 
sample_maxP = pe_samples[np.argmax(log_prob)]
sample_maxP_dict = get_param_dict(sample_maxP)
with open('GW190521_maxP.json', 'w') as f:
    json.dump(sample_maxP_dict, f)
    
# Locateand save the maximum likelihood sample 
sample_maxL = pe_samples[np.argmax(pe_samples['log_likelihood'])]
sample_maxL_dict = get_param_dict(sample_maxL)
with open('GW190521_maxL.json', 'w') as f:
    json.dump(sample_maxL_dict, f)
      
# 10 fair fraws    
fair_draws = np.random.choice(pe_samples, size=10)
for i, f in enumerate(fair_draws): 
    sample_dict = get_param_dict(f)
    with open(f'GW190521_fairdraw_{int(i)}.json', 'w') as f:
        json.dump(sample_dict, f)