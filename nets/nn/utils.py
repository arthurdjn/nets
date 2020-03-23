def info_layer(network, tab=''):
    string = ""
    name = network.__class__.__name__
    string += f'{tab}{name}:'
    for (key, param) in network.parameters().items():
        string += f' {key}: {param.shape}\t'
    string += '\n'
    return string