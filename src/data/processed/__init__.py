from src.utils.registry import Registry

PROCESSED_LOADER_REGISTRY = Registry("processed_loader")


def register_processed_loader(name):
    def decorator(builder):
        PROCESSED_LOADER_REGISTRY.register(name, builder)
        return builder
    return decorator


def create_processed_loader(cfg_loader, **deps):
    # Ensure built-in loaders are registered before lookup
    _load_default_processed_loaders()
    return PROCESSED_LOADER_REGISTRY.build(cfg_loader.name, cfg_loader, **deps)


_DEFAULTS_LOADED = False


def _load_default_processed_loaders():
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return
    # Import modules that register loaders via decorators.
    from src.data.processed import gamenet_loader  # noqa: F401

    _DEFAULTS_LOADED = True
