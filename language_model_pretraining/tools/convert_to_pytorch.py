import pickle
import torch
import numpy as np

# def jax2pytorch(jax_params):
#     torch_params = {}
#     torch_params['input.weight'] = torch.Tensor(np.array(jax_params['Dense_0']['kernel']).T)
#     torch_params['input.bias'] = torch.Tensor(np.array(jax_params['Dense_0']['bias']))
#     torch_params['linear_0.weight'] = torch.Tensor(np.array(jax_params['Dense_1']['kernel']).T)
#     torch_params['linear_0.bias'] = torch.Tensor(np.array(jax_params['Dense_1']['bias']))
#     torch_params['output.weight'] = torch.Tensor(np.array(jax_params['Dense_2']['kernel']).T)
#     torch_params['output.bias'] = torch.Tensor(np.array(jax_params['Dense_2']['bias']))
#     return torch_params

# with open('model_params.pkl', 'rb') as f:
#     jax_params = pickle.load(f)

# # jax_params --> params of Lopt in jax
# torch_params = jax2pytorch(jax_params['params'])
# # torch.save(torch_params, 'model_params_torch.pth')

def jax2pytorch(jax_params):
    all_dicts = {}
    decays = {}
    decays['momentum_decays'] = torch.Tensor(np.array(jax_params['momentum_decays']))
    decays['rms_decays'] = torch.Tensor(np.array(jax_params['rms_decays']))
    decays['adafactor_decays'] = torch.Tensor(np.array(jax_params['adafactor_decays']))
    
    # Create state_dict in the format expected by the model
    torch_params = {}
    torch_params['network.input.weight'] = torch.Tensor(np.array(jax_params['nn']['~']['w0']).T)
    torch_params['network.input.bias'] = torch.Tensor(np.array(jax_params['nn']['~']['b0']))
    torch_params['network.linear_0.weight'] = torch.Tensor(np.array(jax_params['nn']['~']['w1']).T)
    torch_params['network.linear_0.bias'] = torch.Tensor(np.array(jax_params['nn']['~']['b1']))
    torch_params['network.output.weight'] = torch.Tensor(np.array(jax_params['nn']['~']['w2']).T)
    torch_params['network.output.bias'] = torch.Tensor(np.array(jax_params['nn']['~']['b2']))
    
    all_dicts = dict(decays=decays, state_dict=torch_params)
    return all_dicts


to_convert = "/home/mila/b/benjamin.therien/pylo_install/pylo-examples/gpt-training/opt_ckpt/mulo_md.pickle"
to_save = to_convert.replace(".pickle",'.pt')
with open(to_convert, "rb") as f:
    jax_params = pickle.load(f)

# jax_params --> params of Lopt in jax
torch_params = jax2pytorch(jax_params)
torch.save(torch_params, to_save)