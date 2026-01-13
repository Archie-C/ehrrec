from src.data.splitting.GAMENetSplitter import GAMENETSplitter

def create_splitter(cfg_splitter):
    name = cfg_splitter.name
    
    if name == "gamenet":
        return GAMENETSplitter()
    
    raise ValueError(f"Unknown splitter: {name}")