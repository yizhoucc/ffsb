import torch
import torch.nn as nn

class PyTorchMlp(nn.Module):  

  def __init__(self, n_inputs=29, n_actions=2):
      nn.Module.__init__(self)

      self.fc1 = nn.Linear(n_inputs, 32)
      self.fc2 = nn.Linear(32, 64)      
      self.fc3 = nn.Linear(64, n_actions)      
      self.activ_fn = nn.ReLU()
      self.out_activ_fn=nn.Tanh()


  def forward(self, x):
      x = self.activ_fn(self.fc1(x))
      x = self.activ_fn(self.fc2(x))
      x = self.out_activ_fn(self.fc3(x))
      return x


def copy_mlp_weights(baselines_model):
  torch_mlp = PyTorchMlp(n_inputs=29, n_actions=2)
  model_params = baselines_model.get_parameters()
  
  policy_keys = [key for key in model_params.keys() if "pi" in key]
  policy_params = [model_params[key] for key in policy_keys]
    
  for (torch_key, pytorch_param), key, policy_param in zip(torch_mlp.named_parameters(), policy_keys, policy_params):
    # if "pi" in key:
    #     param = torch.from_numpy(policy_param).t()    # not very precise, to 4 digit?
    param = torch.from_numpy(policy_param)
    #Copies parameters from baselines model to pytorch model
    # print(torch_key, key)
    # print(pytorch_param.shape, param.shape, policy_param.shape)
    pytorch_param.data.copy_(param.data.clone().t())
    
  return torch_mlp


if __name__=="__main__":

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

    baselines_mlp_model = DDPG.load("DDPG_ff_mlp_32")
    for key, value in baselines_mlp_model.get_parameters().items():
        if "pi" in key and "model" in key:
            print(key, value.shape)

        

    torch_model = copy_mlp_weights(baselines_mlp_model)

    for a in torch_model.named_parameters():
        print(a[0],a[1].shape)

    # eveal
    b=torch.ones(1,29)
    print(torch_model(b))
    print(baselines_mlp_model.predict(b))


    print('done')