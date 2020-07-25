import torch
import torch.nn as nn

class PyTorchMlp(nn.Module):  

  def __init__(self, n_inputs=29, n_actions=2,layers=[256,256,64,32],act_fn=nn.functional.relu):
      nn.Module.__init__(self)
      self.num_layers=len(layers)
      self.fc1 = nn.Linear(n_inputs, layers[0])
      self.fc2 = nn.Linear(layers[0], layers[1])  
      if self.num_layers==2:   
        self.fc3 = nn.Linear(layers[1], n_actions) 
      else:     
        self.fc3 = nn.Linear(layers[1], layers[2])
        self.fc4 = nn.Linear(layers[2], layers[3])
        self.fc5 = nn.Linear(layers[3], n_actions)
      self.activ_fn = act_fn
      self.out_activ_fn=nn.Tanh()
      self.name=None

  def forward(self, x):
      x = self.activ_fn(self.fc1(x))
      x = self.activ_fn(self.fc2(x))
      if self.num_layers==2:
        x = self.out_activ_fn(self.fc3(x))
      else:
        x = self.activ_fn(self.fc3(x))    
        x = self.activ_fn(self.fc4(x))    
        x = self.out_activ_fn(self.fc5(x))
      return x


def copy_mlp_weights(baselines_model,layers=[64,64],act_fn=nn.functional.relu):
  torch_mlp = PyTorchMlp(n_inputs=29, n_actions=2,layers=layers,act_fn=act_fn)
  model_params = baselines_model.get_parameters()
  
  policy_keys = [key for key in model_params.keys() if "target/pi" in key]
  policy_params = [model_params[key] for key in policy_keys]
    
  for (torch_key, pytorch_param), key, policy_param in zip(torch_mlp.named_parameters(), policy_keys, policy_params):
    # if "pi" in key:
    #     param = torch.from_numpy(policy_param).t()    # not very precise, to 4 digit?
    param = torch.from_numpy(policy_param)
    # Copies parameters from baselines model to pytorch model
    # print(torch_key, key)
    # print(pytorch_param.shape, param.shape, policy_param.shape)
    pytorch_param.data.copy_(param.data.clone().t())
    
  return torch_mlp


if __name__=="__main__":

    # import os
    # import warnings
    # warnings.filterwarnings('ignore')
    # agent
    from stable_baselines import DDPG
    torch.set_printoptions(precision=8)
    # env
    # from FireflyEnv import ffenv
    # from Config import Config
    # arg=Config()
    # from Inverse_Config import Inverse_Config


    # env =ffenv.FireflyEnv(arg)
    # obs = env.reset()

    baselines_mlp_model = DDPG.load("DDPG_selu")
    for key, value in baselines_mlp_model.get_parameters().items():
        if "target/pi" in key:
            print(key, value.shape)
            if 'target/pi/fc0/bias' in key:
              print(value)

        
    # torch_model = copy_mlp_weights(baselines_mlp_model)

    torch_model = copy_mlp_weights(baselines_mlp_model,layer1=128,layer2=128)

    for key, param in torch_model.named_parameters():
        print(key,param.shape)
        if 'fc1.bias' in key:
          print(param)

    # eveal
    b=torch.ones(1,29)
    print(torch_model(b))
    print(baselines_mlp_model.predict(b))


    print('done')