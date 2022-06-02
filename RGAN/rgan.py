import torch 
import torch.nn as nn
from utils import batch_generator


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, feat):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers = 1, batch_first = True)
        self.linear = nn.Linear(hidden_dim, feat)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.tanh(x)
        return x 

class Discriminator(nn.Module):
    def __init__(self, feat, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(feat, hidden_dim, num_layers = 3, bidirectional=True, batch_first = True)
        self.linear = nn.Linear(hidden_dim*2, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x) 
        return x


def rgan(ori_data, parameters):
    generator = Generator(parameters['latent_dim'], parameters['hidden_dim'], parameters['feat']).to(parameters['device'])
    discriminator = Discriminator(parameters['feat'], parameters['hidden_dim']).to(parameters['device'])

    
    # Optimizers for the models, Adam optimizer with learning rate = 0.001 for the generator and SGD with learning rate of 0.1 for the discriminator.
    gen_optim = torch.optim.Adam(generator.parameters(), lr=0.001)
    disc_optim = torch.optim.SGD(discriminator.parameters(), lr = 0.1)

    # Batch generator, it keeps on generating batches of data.
    data_gen = batch_generator(ori_data, parameters)
    # BCE with logits
    criterion = torch.nn.BCEWithLogitsLoss()

    for step in range(parameters['iterations']):
        for disc_step in range(parameters['disc_extra_steps']):
            """
            Discriminator training.
            
            - Generate fake data from the generator.
            - Train the discriminator on the real data and the fake data.

            Note: Make sure to detach the variable from the graph to prevent backpropagation. 
                in this case, it is the synthetic data, (generator(noise)).
            """
            # Get the real batch data, and synthetic batch data. 
            bdata = data_gen.__next__() 
            noise = torch.randn(parameters['batch_size'], parameters['seq_len'], parameters['latent_dim']).to(parameters['device'])

            fake = generator(noise).detach() 
            fake_dscore = discriminator(fake)
            true_dscore = discriminator(bdata)

            # Compute the loss for the discriminator, and backpropagate the gradients.
            dloss = criterion(fake_dscore, torch.zeros_like(fake_dscore)) + criterion(true_dscore, torch.ones_like(true_dscore))
            disc_optim.zero_grad()
            dloss.backward()
            disc_optim.step()

        noise = torch.randn(parameters['batch_size'], parameters['seq_len'], parameters['latent_dim']).to(parameters['device'])
        fake = generator(noise) 
        fake_dscore = discriminator(fake)

        # Compute the loss for the generator, and backpropagate the gradients.
        gloss = criterion(fake_dscore, torch.ones_like(fake_dscore))

        gen_optim.zero_grad()
        gloss.backward()
        gen_optim.step()
        
        print('[Step {}; L(G): {}; L(D): {}]'.format(step, dloss, gloss))


    torch.save(generator.state_dict(), parameters['pre_train_path'] + 'generator.mdl')
    torch.save(discriminator.state_dict(), parameters['pre_train_path'] + 'discriminator.mdl')

    noise = torch.randn(ori_data.shape[0], parameters['seq_len'], parameters['latent_dim']).to(parameters['device'])
    synthetic_samples = generator(noise).detach().cpu().numpy()
    return synthetic_samples



