models_registry = {}

def register_model(version):
    def _register_model(func):
        global models_registry
        models_registry[f'get_{version}_{func.__name__}'] = func
        return func
    return _register_model

# def register_model(version):
#     def decorator(func):
#         print(version)
#         def wrapper(*args, **kwargs):
#             global models_registry
#             models_registry[f'get_{version}_{func.__name__}'] = func
#         return wrapper
#     return decorator

def get_model_fn(version, backbone):
    global models_registry
    # print(models_registry)
    return models_registry[f'get_{version}_{backbone}']