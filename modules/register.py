module_registry = {}

def register(func):
    global module_registry
    module_registry[func.__name__] = func
    return func

def get_pool_search_module(name):
    global module_registry
    return module_registry[name]