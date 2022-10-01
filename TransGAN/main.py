import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# 1. Transformer based TimeGAN model
from transgan import transgan
# 2. Data loading
from data_loading import real_data_loading
# 3. Utils
from utils import Parameters
from shared.component_logger import component_logger as logger

#set the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Data loading
data_path = "data/"
dataset = "stock"
path_real_data = "data/" + dataset + "_data.csv"

#parameters

params = Parameters()
params.dataset = dataset
params.data_path = "data/" + params.dataset + "_data.csv"
params.model_save_path = "checkpoints/" + params.dataset
params.seq_len = 64
params.batch_size = 32
params.max_steps = 10000
params.gamma = 1.0
params.save_model = True
params.print_every = 100
params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params.save_synth_data = False

#preprocessing the data.

"""
Method: real_data_loading()
---------------------------------------------------------------------------------------------------------------------
    - Loads the data from the path.
    - Scales the data using a min-max scaler.
    - Slices the data into windows of size seq_len.
    - Shuffles the data randomly, and returns it.
"""
ori_data, (minimum, maximum) = real_data_loading(path_real_data, params.seq_len)

params.input_size = ori_data[0].shape[1]
params.hidden_size = 24

logger.log('Preprocessing Complete!')
   
with open(data_path + params.dataset + '_real_data.npy', 'wb') as f:
    np.save(f, np.array(ori_data))

logger.log("Saved real data!")

# Run TimeGAN
"""
Method: timegan()
---------------------------------------------------------------------------------------------------------------------
    - Runs the timegan model.
"""
generated_data = transgan(ori_data, params)  

# # Renormalization
# generated_data = generated_data*maximum
# generated_data = generated_data + minimum 

with open(data_path + params.dataset + '_synthetic_data.npy', 'wb') as f:
    np.save(f, np.array(generated_data))