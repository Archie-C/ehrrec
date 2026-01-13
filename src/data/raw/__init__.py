from src.utils.registry import Registry

RAW_LOADER_REGISTRY = Registry("raw_loader")


def register_raw_loader(name):
    def decorator(builder):
        RAW_LOADER_REGISTRY.register(name, builder)
        return builder
    return decorator


def create_raw_loader(cfg_dataset, **deps):
    _load_default_raw_loaders()
    return RAW_LOADER_REGISTRY.build(cfg_dataset.name, cfg_dataset, **deps)


_RAW_DEFAULTS_LOADED = False


def _load_default_raw_loaders():
    global _RAW_DEFAULTS_LOADED
    if _RAW_DEFAULTS_LOADED:
        return
    from src.data.raw import mimic3_loader  # noqa: F401

    _RAW_DEFAULTS_LOADED = True
