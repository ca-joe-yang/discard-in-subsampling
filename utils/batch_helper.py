from torch import Tensor

def list_to_device(
    inputs: list[Tensor], 
    device
):
    r'''Move a list of Tensor to `device`
    '''
    outputs = [ t.to(device) for t in inputs ]
    return outputs

def dict_to_device(
    inputs: dict[str, Tensor], 
    device
):
    r'''Move a dict of Tensor to `device`
    '''
    outputs = { key: item.to(device) for key, item in inputs.items() }
    return outputs