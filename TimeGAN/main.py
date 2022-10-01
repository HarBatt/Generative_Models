import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import TimeGAN
# 2. Data loading
from data_loading import real_data_loading
# 3. Utils
from utils import Parameters
from shared.component_logger import component_logger as logger
import os


# Data loading
params = Parameters()
data_path = "data"
params.dataset = "stock"
params.data_path = os.path.join(data_path, "{}_data.csv".format(params.dataset))
params.model_save_path = os.path.join("checkpoints", params.dataset)

#Hyper-parameters
params.seq_len = 24
params.batch_size = 128
params.max_steps = 1000
params.save_model = True
params.print_every = 1

#set the device 
params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#preprocessing the data.

"""
Method: real_data_loading()
---------------------------------------------------------------------------------------------------------------------
    - Loads the data from the path.
    - Scales the data using a min-max scaler.
    - Slices the data into windows of size seq_len.
    - Shuffles the data randomly, and returns it.
"""
ori_data, (minimum, maximum) = real_data_loading(params.data_path, params.seq_len)

params.input_size = ori_data[0].shape[1]
params.latent_dim = params.input_size # For the sake of simplicity, we assume the latent dimension is the same as the input dimension.
params.hidden_size = 24
params.num_layers = 3
params.gamma = 1
params.disc_extra_steps = 1 # The number of times the discriminator is trained more than the generator.

logger.log("Preprocessing Complete!")
   

with open(os.path.join(data_path, "{}_real_data.npy".format(params.dataset)), 'wb') as f:
    np.save(f, np.array(ori_data))

logger.log("Saved real data!")


"""
Method: train()
---------------------------------------------------------------------------------------------------------------------
    - Trains the RGAN model, and saves model weights.
"""

timegan = TimeGAN(params)
timegan.train(ori_data)  
