import torch
import matplotlib.pyplot as plt
import numpy as np

#creating linear beta schedule
def make_beta_schedule():
    betas= torch.linspace(0.0001, 0.02, 1000)
    return betas

#creating alpha schedule
def make_alpha_schedule():
    betas = make_beta_schedule()
    alphas = 1- betas
    alpha_bars= torch.ones((len(betas)))
    for j in range(len(betas)):
        if j == 0:
            alpha_bars[j]= alphas[0]
        else:
            alpha_bars[j]=alpha_bars[j-1]*alphas[j]
    return alphas, alpha_bars

def make_variance_schedule():
    alphas, alpha_bars= make_alpha_schedule()
    variances= (1-alpha_bars)** -0.5
    return variances

#selecting noising alphas 
def select_alphas(timesteps: torch.Tensor):
    alphas,alpha_bars= make_alpha_schedule()
    selected_alpha_bars= torch.gather(alpha_bars,0,timesteps)
    selected_alphas=torch.gather(alphas,0,timesteps)
    return selected_alpha_bars,selected_alphas

def select_variances(timesteps: torch.Tensor):
    variances= make_variance_schedule
    selected_variances= torch.gather(variances,0,timesteps)
    return selected_variances

#takes in clean image and timesteps and noises the image
def noise_images(x0:torch.Tensor, timestep:torch.Tensor):
    '''
    Args:
    x0 - tensor of dimension (batch_size,channels,height,width)
    timestep - tensor of dimension (batch_size,) 
    '''
    noise = torch.rand_like(x0)
    variances = select_variances(timesteps=timestep)
    alpha_bars = select_alphas(timesteps=timestep)

    #forward pass
    noised_images= alpha_bars * x0 + variances * noise 
    return noised_images

def make_visualization(x:torch.Tensor):
    '''
    Args:
    x-tensor of dimension (batch_size,channels,height, width)
    '''
    sample= x[:4]
    plt.imshow(np.transpose(sample.cpu().numpy(),(1,2,0)))
    plt.axis('off')
    plt.show()

    










            




