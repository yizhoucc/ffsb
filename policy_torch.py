import torch
import torch.nn as nn

class PyTorchMlp(nn.Module):  

  def __init__(self, n_inputs=4, n_actions=2):
      nn.Module.__init__(self)

      self.fc1 = nn.Linear(n_inputs, 64)
      self.fc2 = nn.Linear(64, 64)      
      self.fc3 = nn.Linear(64, n_actions)      
      self.activ_fn = nn.Tanh()
      self.out_activ = nn.Softmax(dim=0)

  def forward(self, x):
      x = self.activ_fn(self.fc1(x))
      x = self.activ_fn(self.fc2(x))
      x = self.out_activ(self.fc3(x))
      return x


def copy_mlp_weights(baselines_model):
  torch_mlp = PyTorchMlp(n_inputs=29, n_actions=2)
  model_params = baselines_model.get_parameters()
  
  policy_keys = [key for key in model_params.keys() if "pi" in key]
  policy_params = [model_params[key] for key in policy_keys]
    
  for (torch_key, pytorch_param), key, policy_param in zip(torch_mlp.named_parameters(), policy_keys, policy_params):
    param = torch.from_numpy(policy_param)    
    #Copies parameters from baselines model to pytorch model
    print(torch_key, key)
    print(pytorch_param.shape, param.shape, policy_param.shape)
    pytorch_param.data.copy_(param.data.clone().t())
    
  return torch_mlp


# if __name__=="__main__":

    import os
    import warnings
    warnings.filterwarnings('ignore')
    from inverse_model import Dynamic, Inverse
    # agent
    from stable_baselines import DDPG
    # env
    from FireflyEnv import ffenv
    from Config import Config
    arg=Config()
    from Inverse_Config import Inverse_Config


    env =ffenv.FireflyEnv(arg)
    obs = env.reset()

    baselines_mlp_model = DDPG.load("DDPG_ff_mlp")
    for key, value in baselines_mlp_model.get_parameters().items():
        print(key, value.shape)

    torch_model = copy_mlp_weights(baselines_mlp_model)