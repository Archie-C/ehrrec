import hydra
from omegaconf import DictConfig
from torch import manual_seed as torch_seed
from torch import device as torch_device
from torch import cuda
from numpy.random import seed as np_seed
from random import seed as random_seed
from torch.backends import cudnn
from torch.nn import Module


from src.evaluation.evaluator_factory import create_evaluator
from src.data.processed import create_processed_loader
from src.data.splitting.splitter_factory import create_splitter
from src.models.model_factory import create_model
from src.data.preprocessors.preprocessor_factory import create_preprocessor
from src.data.raw import create_raw_loader
from src.data.model_inputs.factory import create_model_input_builder
from src.training.trainer_factory import create_trainer



@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.run.mode in ["full", "process"]:
        raw_loader = create_raw_loader(cfg.dataset)
        raw_data = raw_loader.load(cfg.dataset.paths, cfg.dataset.kwargs)

        preprocessor = create_preprocessor(cfg.preprocessor, log_level=cfg.logging.level)
        processed_data = preprocessor.run(raw_data)
        preprocessor.save(processed_data)

        # If process-only, stop here
        if cfg.run.mode == "process":
            return
        
    torch_seed(cfg.run.seed)
    np_seed(cfg.run.seed)
    random_seed(cfg.run.seed)
    cuda.manual_seed_all(cfg.run.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    dataset_context = create_processed_loader(cfg.preprocessor).load()
    splitter = create_splitter(cfg.splitter)
    train_splits, val_splits, test_splits = splitter.split(dataset_context)
    device = torch_device("cuda" if cuda.is_available() else "cpu")
    model = create_model(cfg.model, device=device)
    if isinstance(model, Module):
        model = model.to(device)
    
    voc_size = dataset_context.vocab_sizes()
    
    model_input_builder = create_model_input_builder(
        cfg.model_inputs,
        log_level=cfg.logging.level,
        run_mode=cfg.run.mode,
    )
    train_data, val_data, test_data = model_input_builder.run(
        source=dataset_context.name,
        context=dataset_context,
        train_data=train_splits,
        val_data=val_splits,
        test_data=test_splits,
        voc_size=voc_size,
    )
    
    
    if cfg.run.mode == "full" or cfg.run.mode == "train":
        trainer = create_trainer(cfg.trainer)
        trainer.train(model, train_data, val_data, context=dataset_context)
        
        if cfg.run.mode == "train":
            return
    
    if cfg.run.mode == "full" or cfg.run.mode == "test":
        if hasattr(model, "load_memory"):
            model.load_memory(cfg.run.model, map_location=device)
            extras = None
        else:
            extras = model.load(cfg.run.model)
        evaluator = create_evaluator(cfg.evaluator)
        results = evaluator.evaluate(model, test_data, context=dataset_context)
        
        if cfg.run.mode == "test":
            return
        

if __name__ == "__main__":
    main()
