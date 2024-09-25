models_registry = {}

def register_model(version):
    def _register_model(func):
        global models_registry
        models_registry[f'get_{version}_{func.__name__}'] = func
        return func
    return _register_model


def get_model_fn(version, backbone):
    global models_registry
    return models_registry[f'get_{version}_{backbone}']