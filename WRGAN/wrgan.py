import torch
from utils import batch_generator
from shared.component_logger import component_logger as logger
from modules import Generator, Discriminator


class RecurrentWGAN:
    def __init__(self, params):
        self.params = params 

        # 1. Create the generator and the discriminator
        self.generator = Generator(params.latent_dim, params.input_size, params.num_layers).to(params.device)
        self.discriminator = Discriminator(params.input_size, params.hidden_size, params.num_layers).to(params.device)

        # Optimizers for the models, Adam optimizer with learning rate = 0.001 for the generator and SGD with learning rate of 0.1 for the discriminator.
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr= 0.001, betas=(0.5, 0.9))
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr= 0.001, betas=(0.5, 0.9))

    def save_model(self, model, step, path):
        """
        params:
            model: model to save
            step: step of the model
            path: path to save the model
        """
        if model =="generator":
            torch.save({
                'step': step,
                'model_state_dict': self.generator.state_dict(),
                'optimizer_state_dict': self.gen_optim.state_dict(),
            }, path)
        
        elif model=="discriminator":
            torch.save({
                'step': step,
                'model_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.disc_optim.state_dict(),
            }, path)
        
        else:
            raise ValueError("Invalid model or path name")

    def load_model(self, model, path):
        """
        Load model from the path.
        params:
            model: model name
            path: path to the model
        """
        if model =="generator":
            checkpoint = torch.load(path)
            self.generator.load_state_dict(checkpoint['model_state_dict'])
            self.gen_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        elif model=="discriminator":
            checkpoint = torch.load(path)
            self.discriminator.load_state_dict(checkpoint['model_state_dict'])
            self.disc_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        else:
            raise ValueError("Invalid model or path name")


    def generate_synthetic_data(self, num_samples):
        """
            params: 
                num_samples: number of samples to generate
            return:
                x: generated data
        """
        params = self.params
        noise = torch.randn(num_samples, params.seq_len, params.latent_dim).to(params.device)
        x_hat = self.generator(noise) 
        synthetic_samples = x_hat.detach().cpu().numpy()
        return synthetic_samples




    def train(self, ori_data):
        """
            Train the WRGAN model with gradient penalty.
        """
        params = self.params
    
        # Batch generator, it keeps on generating batches of data.
        data_gen = batch_generator(ori_data, params)

        with torch.backends.cudnn.flags(enabled=False):
            
            for step in range(params.max_steps):
                for disc_step in range(params.disc_extra_steps):
                    """
                    Discriminator training.
                    
                    - Generate fake data from the generator.
                    - Train the discriminator on the real data and the fake data.

                    Note: Make sure to detach the variable from the graph to prevent backpropagation. 
                        in this case, it is the synthetic data, (generator(noise)).
                    """
                    # Get the real batch data, and synthetic batch data. 
                    bdata = data_gen.__next__() 
                    noise = torch.randn(params.batch_size, params.seq_len, params.latent_dim).to(params.device)

                    fake = self.generator(noise).detach() 
                    fake_dscore = self.discriminator(fake)
                    true_dscore = self.discriminator(bdata)

                    # Compute gradient penalty, as per the paper.
                    epsilon = torch.rand(params.batch_size, 1, 1).to(params.device)
                    # x_hat represents and interpolated sample between real and fake data.
                    x_hat = (epsilon * bdata + (1 - epsilon) * fake).requires_grad_(True)  
                    dscore_hat = self.discriminator(x_hat)
                    gradients = torch.autograd.grad(outputs=dscore_hat, inputs=x_hat, grad_outputs=torch.ones(dscore_hat.size()).to(params.device), create_graph=True, retain_graph=True, only_inputs=True)[0]

                    # Compute the penalty.
                    gp = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-10)
                    gp = torch.mean((gp - 1)**2)

                    # Compute the loss.
                    wasserstein_distance = torch.mean(true_dscore) - torch.mean(fake_dscore)
                    penalty = params.gp_lambda * gp
                    dloss = -wasserstein_distance + penalty
                    self.disc_optim.zero_grad()
                    dloss.backward()
                    self.disc_optim.step()
                
                """Generator training."""
                # Generate fake data from the generator.
                noise = torch.randn(params.batch_size, params.seq_len, params.latent_dim).to(params.device)
                fake = self.generator(noise) 
                fake_dscore = self.discriminator(fake)

                # Compute the loss for the generator, and backpropagate the gradients.
                gloss = -1.0*torch.mean(fake_dscore)
                self.gen_optim.zero_grad()
                gloss.backward()
                self.gen_optim.step()

                if step % params.print_every== 0:
                    self.save_model("generator", step, params.model_save_path + '_generator.pth')
                    self.save_model("discriminator", step, params.model_save_path + '_discriminator.pth')
                    logger.log('[Step {}; L(C): {}; L(G): {}; dist(W): {}]'.format(step, dloss, gloss, wasserstein_distance))

        logger.log("Finish Training")




