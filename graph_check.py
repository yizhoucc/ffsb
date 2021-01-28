# visualize graph

import os
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"
from graphviz import Digraph
import torch
from torch.autograd import Variable


def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph
    
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)
    
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    
    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

# baselines_mlp_model = TD3.load('trained_agent//TD_action_cost_700000_8_19_21_56.zip')
# agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[128,128],n_inputs=31)

# model=agent
# input_size=(1,31)
def checker(model, input_size):
    inputs = torch.randn(input_size)
    y = model(Variable(inputs))
    g = make_dot(y, model.state_dict())
    g.view()

params={'theta':theta_estimation}
env.reset(theta=theta_estimation)

var=agent(env.decision_info)
env(var,task_param=theta_estimation)

g = make_dot(var, params)
g = make_dot(env.decision_info, params)
g = make_dot(loss, params)

g.view()

if __name__ == "__main__":  
    from torchvision import models
    inputs = torch.randn(1,3,224,224)
    resnet18 = models.resnet18()
    y = resnet18(Variable(inputs))
    # print(y)

    g = make_dot(y, resnet18.state_dict())
    g.view()