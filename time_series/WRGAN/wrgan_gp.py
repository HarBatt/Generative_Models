import torch 
import torch.nn as nn
from utils import batch_generator


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, feat):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers = 3, batch_first = True)
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


def wrgan_gp(ori_data, parameters):
    generator = Generator(parameters['latent_dim'], parameters['hidden_dim'], parameters['feat']).to(parameters['device'])
    discriminator = Discriminator(parameters['feat'], parameters['hidden_dim']).to(parameters['device'])

    
    # Optimizers for the models, Adam optimizer with learning rate = 0.001 for the generator and SGD with learning rate of 0.1 for the discriminator.
    gen_optim = torch.optim.Adam(generator.parameters(), lr=parameters['lr_g'], betas=(0.5, 0.9))
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=parameters['lr_d'], betas=(0.5, 0.9))
    
    # Batch generator, it keeps on generating batches of data.
    data_gen = batch_generator(ori_data, parameters)
    with torch.backends.cudnn.flags(enabled=False):

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

                # Compute gradient penalty, as per the paper.
                epsilon = torch.rand(parameters['batch_size'], 1, 1).to(parameters['device'])
                # x_hat represents and interpolated sample between real and fake data.
                x_hat = (epsilon * bdata + (1 - epsilon) * fake).requires_grad_(True)  
                dscore_hat = discriminator(x_hat)
                gradients = torch.autograd.grad(outputs=dscore_hat, inputs=x_hat, grad_outputs=torch.ones(dscore_hat.size()).to(parameters['device']), create_graph=True, retain_graph=True, only_inputs=True)[0]

                # Compute the penalty.
                gp = torch.sqrt(torch.sum(gradients** 2, dim=1) + 1e-10)
                gp = torch.mean((gp-1)**2)

                # Compute the loss.
                wasserstein_distance = torch.mean(true_dscore) - torch.mean(fake_dscore)
                penalty = parameters['gp_lambda'] * gp
                dloss = -wasserstein_distance + penalty
                disc_optim.zero_grad()
                dloss.backward()
                disc_optim.step()
            
            """
            Generator training.
            """
            # Generate fake data from the generator.
            noise = torch.randn(parameters['batch_size'], parameters['seq_len'], parameters['latent_dim']).to(parameters['device'])
            fake = generator(noise) 
            fake_dscore = discriminator(fake)

            # Compute the loss for the generator, and backpropagate the gradients.
            gloss = -1.0*torch.mean(fake_dscore)
            gen_optim.zero_grad()
            gloss.backward()
            gen_optim.step()
            print('[Step {}; L(C): {}; L(G): {}; Wass_Distance: {}]'.format(step + 1, dloss, gloss, wasserstein_distance))


    torch.save(generator.state_dict(), parameters['pre_train_path'] + 'generator.mdl')
    torch.save(discriminator.state_dict(), parameters['pre_train_path'] + 'discriminator.mdl')

    noise = torch.randn(ori_data.shape[0], parameters['seq_len'], parameters['latent_dim']).to(parameters['device'])
    synthetic_samples = generator(noise).detach().cpu().numpy()
    return synthetic_samples



