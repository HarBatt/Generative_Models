import torch 
import numpy as np
from utils import batch_generator
from shared.component_logger import component_logger as logger
from modules import Generator, Discriminator


class RecurrentGAN:
    """
       Recurrent Generative Adversarial Network:
       -----------------------------------------------------------
       Loosely based on the open source implementation of the RCGAN

       Reference:
       Esteban, Cristóbal, Stephanie L. Hyland, and Gunnar Rätsch. "Real-valued (medical) time series generation with recurrent conditional gans." 
       arXiv preprint arXiv:1706.02633 (2017).
    """
    def __init__(self, params):
        self.params = params

        # 1. Create the generator and the discriminator
        self.generator = Generator(params.latent_dim, params.hidden_size, params.input_size, params.num_layers).to(params.device)
        self.discriminator = Discriminator(params.input_size, params.hidden_size, params.num_layers).to(params.device)

        # 2. Create the optimizers, Stochastic Gradient Descent optimizer with learning rate = 0.001 for the generator and SGD with learning rate of 0.1 for the discriminator.
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.disc_optim = torch.optim.SGD(self.discriminator.parameters(), lr = 0.1)

        logger.log("Traininable Parameters in Generator: {:,}".format(self.parameters_count(self.generator)))
        logger.log("Traininable Parameters in Discriminator: {:,}".format(self.parameters_count(self.discriminator)))

    
    def parameters_count(self, model):
        """
        params:
            model: A neural network architecture to count trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    

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
            Train the RGAN model.
        """
        # parameters object for the model
        params = self.params 
        # Batch generator, it keeps on generating batches of data.
        data_gen = batch_generator(ori_data, params)
        # Loss: BCE with logits
        criterion = torch.nn.BCEWithLogitsLoss()

        for step in range(params.max_steps):
            for disc_step in range(params.disc_extra_steps): 
                """
                Discriminator training.
                -----------------------------------------------------------
                
                - Generate fake data from the generator.
                - Train the discriminator on the real data and the fake data.

                """
                # Get the real batch data, and synthetic batch data. 
                bdata = data_gen.__next__() 
                noise = torch.randn(params.batch_size, params.seq_len, params.latent_dim).to(params.device)

                fake = self.generator(noise).detach() 
                fake_dscore = self.discriminator(fake)
                true_dscore = self.discriminator(bdata)

                # Compute the loss for the discriminator, and backpropagate the gradients.
                dloss = criterion(fake_dscore, torch.zeros_like(fake_dscore)) + criterion(true_dscore, torch.ones_like(true_dscore))
                self.disc_optim.zero_grad()
                dloss.backward()
                self.disc_optim.step()

            """
            Generator training.
            -----------------------------------------------------------
            - Generate fake data from the generator.
            - Evaluate fake scores with BCE loss against 1 with discriminator.
            """

            noise = torch.randn(params.batch_size, params.seq_len, params.latent_dim).to(params.device)
            fake = self.generator(noise) 
            fake_dscore = self.discriminator(fake)

            # Compute the loss for the generator, and backpropagate the gradients.
            gloss = criterion(fake_dscore, torch.ones_like(fake_dscore))

            self.gen_optim.zero_grad()
            gloss.backward()
            self.gen_optim.step()

            if step % params.print_every== 0:
                if params.save_model:
                    self.save_model("generator", step, params.model_save_path + '_generator.pth')
                    self.save_model("discriminator", step, params.model_save_path + '_discriminator.pth')

                logger.log('[Step {}; L(D): {}; L(G): {}]'.format(step, np.round(dloss.item(), 4), np.round(gloss.item(), 4)))

        logger.log("Finish Training")

