## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# 1. WRGAN model
from wrgan import RecurrentWGAN
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
#Evaluation of the model, by default can be set to false.
eval_model = False

#parameters

params = Parameters()
params.dataset = dataset
params.data_path = "data/" + params.dataset + "_data.csv"
params.model_save_path = "checkpoints/" + params.dataset
params.seq_len = 24
params.batch_size = 128
params.max_steps = 10000
params.save_model = True
params.print_every = 500
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
params.latent_dim = params.input_size # For the sake of simplicity, we assume the latent dimension is the same as the input dimension.
params.hidden_size = 24
params.num_layers = 3
params.disc_extra_steps = 5 # The number of times the discriminator is trained more than the generator.
params.gp_lambda = 5 # The lambda value for the gradient penalty.

logger.log("Preprocessing Complete!")
   
with open(data_path + params.dataset + '_real_data.npy', 'wb') as f:
    np.save(f, np.array(ori_data))

logger.log("Saved real data!")


"""
Method: train()
---------------------------------------------------------------------------------------------------------------------
    - Trains the wrgan model, and saves model weights.
"""

wrgan = RecurrentWGAN(params)
wrgan.train(ori_data)  
