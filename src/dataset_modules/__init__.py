LOADER_REGISTRY = {}

def register_loader(dataset_name: str):
    """decorator registering a dataset loader class by name."""
    def decorator(cls):
        LOADER_REGISTRY[dataset_name] = cls
        return cls
    return decorator
