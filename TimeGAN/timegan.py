import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from itertools import chain
from utils import batch_generator
from shared.component_logger import component_logger as logger
from modules import Generator, Embedder, Recovery, Supervisor, Discriminator

class Loss:
    """
    Loss functions.
    """
    def __init__(self, params):
        self.params = params

    def loss_e_t0(self, x_tilde, x):
        return F.mse_loss(x_tilde, x)

    def loss_e_0(self, loss_e_t0):
        return torch.sqrt(loss_e_t0) * 10

    def loss_e(self, loss_e_0, loss_s):
        return loss_e_0 + 0.1* loss_s

    def loss_s(self, h_hat_supervise, h):
        return F.mse_loss(h_hat_supervise[:,:-1,:], h[:, 1:, :])

    def loss_g_u(self, y_fake):
        return F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))

    def loss_g_u_e(self, y_fake_e):
        return F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

    def loss_g_v(self, x_hat, x):
        loss_g_v1 = torch.mean(
            torch.abs(torch.sqrt(torch.var(x_hat, 0) + 1e-6) - torch.sqrt(torch.var(x, 0) + 1e-6))
        )
        loss_g_v2 = torch.mean(torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0)))
        return loss_g_v1 + loss_g_v2

    def loss_g(self, loss_g_u, loss_g_u_e, loss_s, loss_g_v):
        return loss_g_u + self.params.gamma * loss_g_u_e + 100 * torch.sqrt(loss_s) + 100 * loss_g_v

    def loss_d(self, y_real, y_fake, y_fake_e):
        loss_d_real = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))
        loss_d_fake = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))
        loss_d_fake_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.zeros_like(y_fake_e))
        return loss_d_real + loss_d_fake + self.params.gamma * loss_d_fake_e

class TimeGAN:
    """
       Time Series Generative Adversarial Network:
       -----------------------------------------------------------
       Implementation based on the open source implementation of the TimeGAN

       Reference:
       Yoon, Jinsung, Daniel Jarrett, and Mihaela Van der Schaar. "Time-series generative adversarial networks." 
       Advances in neural information processing systems 32 (2019).
    """
    def __init__(self, params):
        self.params = params
        self.embedder = Embedder(params.input_size, params.hidden_size, params.hidden_size, params.num_layers).to(params.device)
        self.recovery = Recovery(params.hidden_size, params.hidden_size, params.input_size, params.num_layers).to(params.device)
        self.generator = Generator(params.input_size, params.hidden_size, params.hidden_size, params.num_layers).to(params.device)
        self.supervisor = Supervisor(params.hidden_size, params.hidden_size, params.hidden_size, params.num_layers).to(params.device)
        self.discriminator = Discriminator(params.hidden_size, params.hidden_size, 1, params.num_layers).to(params.device)

        # Optimizers for the models, Adam optimizer
        self.optimizer_er = optim.Adam(chain(self.embedder.parameters(), self.recovery.parameters()))
        self.optimizer_gs = optim.Adam(chain(self.generator.parameters(), self.supervisor.parameters()))
        self.optimizer_d = optim.Adam(self.discriminator.parameters())

    def save_model(self, model, step, path):
        """
        params:
            model: model to save
            step: step of the model
            path: path to save the model
        """
        if model =="embedder":
            torch.save({
                'step': step,
                'model_state_dict': self.embedder.state_dict(),
                'optimizer_state_dict': self.optimizer_er.state_dict(),
            }, path)
        
        elif model=="supervisor":
            torch.save({
                'step': step,
                'model_state_dict': self.supervisor.state_dict(),
                'optimizer_state_dict': self.optimizer_gs.state_dict(),
            }, path)
        
        elif model=="generator":
            torch.save({
                'step': step,
                'model_state_dict': self.generator.state_dict(),
                'optimizer_state_dict': self.optimizer_gs.state_dict(),
            }, path)
        
        elif model=="discriminator":
            torch.save({
                'step': step,
                'model_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.optimizer_d.state_dict(),
            }, path)

        elif model=="recovery":
            torch.save({
                'step': step,
                'model_state_dict': self.recovery.state_dict(),
                'optimizer_state_dict': self.optimizer_er.state_dict(),
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
        if model =="embedder":
            checkpoint = torch.load(path)
            self.embedder.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_er.load_state_dict(checkpoint['optimizer_state_dict'])
        
        elif model=="supervisor":
            checkpoint = torch.load(path)
            self.supervisor.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_gs.load_state_dict(checkpoint['optimizer_state_dict'])
        
        elif model=="generator":
            checkpoint = torch.load(path)
            self.generator.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_gs.load_state_dict(checkpoint['optimizer_state_dict'])

        elif model=="discriminator":
            checkpoint = torch.load(path)
            self.discriminator.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_state_dict'])
        
        elif model=="recovery":
            checkpoint = torch.load(path)
            self.recovery.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_er.load_state_dict(checkpoint['optimizer_state_dict'])

        else:
            raise ValueError("Invalid model or path name")


    def generate_synthetic_data(self, num_samples):
        """
            params: 
                num_samples: number of samples to generate
            return:
                x: generated data
        """
        z = torch.randn(num_samples, self.params.seq_len, self.params.input_size).to(self.params.device)
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)
        x_hat = self.recovery(h_hat)
        
        synthetic_samples = x_hat.detach().cpu().numpy()
        return synthetic_samples

    def train(self, ori_data):
        params = self.params 
        loss = Loss(params)
        
        self.embedder.train()
        self.generator.train()
        self.supervisor.train()
        self.recovery.train()
        self.discriminator.train()
        
        # Batch generator, it keeps on generating batches of data
        data_gen = batch_generator(ori_data, params)

        logger.log("Start Embedding Network Training")

        for step in range(1, params.max_steps + 1):
            # Get the real batch data, and synthetic batch data. 
            x = data_gen.__next__() 
            h = self.embedder(x)
            x_tilde = self.recovery(h)
            loss_e_t0 = loss.loss_e_t0(x_tilde, x)
            loss_e_0 = loss.loss_e_0(loss_e_t0)
            self.optimizer_er.zero_grad()
            loss_e_0.backward()
            self.optimizer_er.step()

            if step % params.print_every == 0:
                self.save_model("embedder", step, params.model_save_path + '_embedder.pth')
                self.save_model("recovery", step, params.model_save_path + '_recovery.pth')
                logger.log("step: "+ str(step)+ "/"+ str(params.max_steps)+ ", loss_e: "+ str(np.round(np.sqrt(loss_e_t0.item()), 4)))

        logger.log("Finish Embedding Network Training")

        logger.log("Start Training with Supervised Loss Only")

        for step in range(1, params.max_steps + 1):
            # Get the real batch data, and synthetic batch data. 
            x = data_gen.__next__()
            z = torch.randn(x.size(0), x.size(1), x.size(2)).to(params.device)

            h = self.embedder(x)
            h_hat_supervise = self.supervisor(h)

            loss_s = loss.loss_s(h_hat_supervise, h)
            self.optimizer_gs.zero_grad()
            loss_s.backward()
            self.optimizer_gs.step()

            if step % params.print_every == 0:
                self.save_model("supervisor", step, params.model_save_path + '_supervisor.pth')
                logger.log("step: "+ str(step)+ "/"+ str(params.max_steps)+ ", loss_s: "+ str(np.round(np.sqrt(loss_s.item()), 4)))

        logger.log("Finish Training with Supervised Loss Only")

        logger.log("Start Joint Training")

        for step in range(1, params.max_steps + 1):
            for _ in range(2):
                x = data_gen.__next__()
                z = torch.randn(x.size(0), x.size(1), x.size(2)).to(params.device)

                h = self.embedder(x)
                e_hat = self.generator(z)
            
                h_hat = self.supervisor(e_hat)
            
                h_hat_supervise = self.supervisor(h)
                x_hat = self.recovery(h_hat)
        
                y_fake = self.discriminator(h_hat)
                y_fake_e = self.discriminator(e_hat)

                loss_s = loss.loss_s(h_hat_supervise, h)
                loss_g_u = loss.loss_g_u(y_fake)
                loss_g_u_e = loss.loss_g_u_e(y_fake_e)
                loss_g_v = loss.loss_g_v(x_hat, x)
                loss_g = loss.loss_g(loss_g_u, loss_g_u_e, loss_s, loss_g_v)
                self.optimizer_gs.zero_grad()
                loss_g.backward()
                self.optimizer_gs.step()

                h = self.embedder(x)
                x_tilde = self.recovery(h)
                h_hat_supervise = self.supervisor(h)

                loss_e_t0 = loss.loss_e_t0(x_tilde, x)
                loss_e_0 = loss.loss_e_0(loss_e_t0)
                loss_s = loss.loss_s(h_hat_supervise, h)
                loss_e = loss.loss_e(loss_e_0, loss_s)
                self.optimizer_er.zero_grad()
                loss_e.backward()
                self.optimizer_er.step()

            x = data_gen.__next__()
            z = torch.randn(x.size(0), x.size(1), x.size(2)).to(params.device)

            h = self.embedder(x)
            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)
            y_fake = self.discriminator(h_hat)
            y_real = self.discriminator(h)
            y_fake_e = self.discriminator(e_hat)

            loss_d = loss.loss_d(y_real, y_fake, y_fake_e)

            # Update the weights of the discriminator only when the loss is large enough
            if loss_d.item() > 0.15:
                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

            if step % params.print_every == 0:
                self.save_model("embedder", step, params.model_save_path + '_embedder.pth')
                self.save_model("supervisor", step, params.model_save_path + '_supervisor.pth')
                self.save_model("generator", step, params.model_save_path + '_generator.pth')
                self.save_model("discriminator", step, params.model_save_path + '_discriminator.pth')
                self.save_model("recovery", step, params.model_save_path + '_recovery.pth')

                logger.log("step: {}/{}, loss_d: {}, loss_g_u: {}, loss_g_v: {}, loss_s: {}, , loss_e_t0: {}".format(step, params.max_steps, np.round(loss_d.item(), 4), np.round(loss_g_u.item(), 4), np.round(loss_g_v.item(), 4), np.round(np.sqrt(loss_s.item()), 4), np.round(np.sqrt(loss_e_t0.item()), 4)))
        
        logger.log("Finish Joint Training")
