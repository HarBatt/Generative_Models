## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# 1. RGAN model
from wrgan_gp import wrgan_gp
# 2. Data loading
from data_loading import real_data_loading
# 3. Metrics
from metrics.visualization_metrics import visualization

"""
Note:
    1) WRGAN-GP returns a sequences of shuffled synthetic data. 
        This synthetic data respects temporal relationships within the sequenences and between the features. 

    2) Make sure to remove the unnecessary attributes from your input file, and also the categorical attributes.
        ** The first column is the time stamp.**

---------------------------------------------------------------------------------------------------------------------
    - It is a simple recurrent neural network (RNN) that generates data from a latent space, and uses Wasserstein distance as the discriminator's loss function. 
    - The synthetic data are sequences of time-series. They preserve temporal structure, and relationships between the features.
    - The sequence length used is 30, a hyper-parameter. 
    - Downstream tasks in which the synthetic data is helpful are:
        - 1) Training models for time-series forecasting. Train the model on 29 time-steps, and predict the next time-step.
        - 2) Training models for time-series anomaly detection.
        - 3) Training models for time-series clustering.
        - 4) Training models for time-series classification.
"""




#set the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Data loading
data_path = "data/"
path_real_data = data_path + 'calaveras_telemetry_multi.csv'
#Evaluation of the model, by default can be set to false.
eval_model = False

#parameters
parameters = dict()

"""
Parameters for the WRGAN-GP model.
---------------------------------------------------------------------------------------------------------------------
1) batch_size: Batch size for training.
2) hidden_dim: Number of hidden units in the recurrent cell.
3) num_layer: Number of layers in the recurrent cell.
4) iterations: Number of epochs to train the model.
5) lr_g: Learning rate for the generator.
6) lr_d: Learning rate for the discriminator.
"""

parameters['pre_train_path'] = "pre_trained_model/" # Path to save the pre-trained model.
parameters['batch_size'] = 64   # Batch size for training.
parameters['hidden_dim'] = 128 #Hidden dim for the LSTM
parameters['iterations'] = 5000 # Number of epochs to train the model.
parameters['latent_dim'] = 10 #Latent dim for the generator
parameters['disc_extra_steps'] = 3 #Extra steps for the discriminator; 5 is the original value from the paper. Use 3 to save time
parameters['feat'] = 6 #Number of features
parameters['device'] = device #Device
parameters['lr_g'] = 0.001 #Learning rate for the generator
parameters['lr_d'] = 0.001 #Learning rate for the discriminator
parameters['seq_len'] = 30 #Length of the sequence to be used, used for slicing the windows.
parameters['gp_lambda'] = 5 #Lambda for the gradient penalty in WGAN-GP


#preprocessing the data.

"""
Method: real_data_loading()
---------------------------------------------------------------------------------------------------------------------
    - Loads the data from the path.
    - Scales the data using a min-max scaler.
    - Slices the data into windows of size seq_len.
    - Shuffles the data randomly, and returns it.

"""
ori_data = real_data_loading(path_real_data, parameters['seq_len'])

print(ori_data.shape)

# 1 + parameters['feat']; 1 due to the timestamp feature. 
assert ori_data.shape[2] == 1 + parameters['feat'], "The number of features in the parameters dictionary is not equal to the number of features in the data."

print('Preprocessing Complete!')

with open(data_path + 'real_data.npy', 'wb') as f:
    np.save(f, np.array(ori_data))

print("Saved real data!")

# Run RGAN
"""
- Overall, (my take on WRGAN): WRGAN with GP is an extension of RGAN with Wasserstein distance and uses the fact that even with no information about the latent space, the model can still learn the temporal structure of the data by using the RNN's previous outputs.  
                            By contrast to RGAN with BCE loss, WRGAN uses eaiser gradient penalty to penalize the difference between the generator and the discriminator.
Method: wrgan_gp()
---------------------------------------------------------------------------------------------------------------------
    - Runs the wrgan_gp model.
"""
generated_data = wrgan_gp(ori_data, parameters)   
print('WRGAN-GP Training Complete!')

with open(data_path + 'synthetic_data.npy', 'wb') as f:
    np.save(f, np.array(generated_data))
