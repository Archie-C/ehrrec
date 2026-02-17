import time
import math
import copy
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Iterable

from src.utils.logging import get_logger
from src.training.BaseTrainer import BaseTrainer
from src.utils.metrics import Metrics

logger = get_logger("Original Models Trainer")

class OriginalModelsTrainer(BaseTrainer):
    def __init__(self, train_config):
        self.__name__ = "Original Models Trainer"
        self.config = train_config
    
    def train(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        run_id: str,
        val_loader: Optional[DataLoader] = None,
        artefacts: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Train function for original models

        Args:
            model (nn.Module): The torch model to train
            train_loader (DataLoader): The training samples to use for training
            val_loader (Optional[DataLoader]): Optional validation samples in the same format as train_loader.
                If provided, validation metrics are computed after each epoch and best model checkpoint is saved.

        Returns:
            Dict[str, Any]:
                - best_model_path: Str of where the best model is saved
                - best_epoch: Epoch number with best validation score
                - best_metrics: Dictionary of best metrics
                - training_time: Total training time in seconds
        """
        logger.info(f"Training starting for model: {repr(model)} on {len(train_loader)} training samples. For {self.config.epochs} epochs. Run_id: {run_id}")
        
        training_start_time = time.time()
        
        # TODO: Add in a more verbose log statement for which requirements aren't met.
        # Checking model requirements to see if the batch_size etc. is correct.
        requirements = model.requirements(config=self.config)
        met = requirements["met"]
        assert met, f"Training requirements for model: {repr(model)} not met."
        
        if val_loader is not None:
            assert artefacts.get("ddi_adj", None) is not None, "Need artefacts['ddi_adj'] to compute validation DDI metrics"
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        logger.info(f"Set rabdin seed to {self.config.seed}")
        
        # Setup model, optimiser and loss function
        model.to(device)
        optimiser = self._optimiser(params=model.parameters())
        logger.info(f"Initialised optimiser: {self.config.optimiser} with lr={self.config.learning_rate}, weight_decay={self.config.weight_decay}")
        
        total_steps = self.config.epochs * len(train_loader)
        warmup_steps = int(0.05 * total_steps)
        scheduler = self._scheduler(
            optimiser=optimiser, 
            warmup_steps=warmup_steps, 
            total_steps=total_steps, 
            min_lr_ratio=self.config.min_lr_ratio
        )
        logger.info("Initialised scheduler: Cosine Scheduler with Warmup")
        
        # Setup the checkpoint directory
        save_dir = Path("saved/" + model.name + "/" + run_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / "best_model.pt"
        logger.info(f"Checkpoints will be saved to: {checkpoint_path}")
        
        # Track best model
        best_score = -float("inf")
        best_epoch = -1
        best_state = None

        history = Dict[str, Any] = {
            "run_id": run_id,
            "best_model_path": checkpoint_path,
            "best_epoch": None,
            "best_metrics": None,
            "validation_metrics": [],
            "train_loss": [],
            "training_time": None
        }

        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            model.train()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            
            epoch_loss_sum = 0.0
            num_batches = 0
            
            # Training
            
            for batch in pbar:
                optimiser.zero_grad(set_to_none=True)
                loss = model.compute_loss(batch, artefacts)
                loss.backward()
                optimiser.step()
                scheduler.step()
                
                epoch_loss_sum += loss
                num_batches += 1
            
            avg_train_loss = epoch_loss_sum / max(num_batches, 1)
            train_epoch_time = time.time() - epoch_start_time
            history["train_loss"].append(avg_train_loss)
            
            logger.info(f"Epoch {epoch} complete in {train_epoch_time:.3f}s | Train Loss={avg_train_loss:.4f}")

            # Validation
            
            if val_loader is not None:
                validation_start_time = time.time()
                
                model.eval()
                pbar = tqdm(val_loader, desc=f"Epoch {epoch} - validation")
                
                num_batches = 0
                val_metrics = Metrics()
                
                for batch in pbar:
                    batch_metrics: Metrics = model.compute_metrics(batch, artefacts)
                    val_metrics.add(batch_metrics)
                    num_batches += 1
                
                val_metrics.avg(max(num_batches, 1))

                history["validation_metrics"].append(val_metrics)
                
                val_time = time.time() - validation_start_time
                
                logger.info(
                    f"Epoch {epoch} Validation ({val_time:.2f}s) | "
                    f"DDI Rate={val_metrics.ddi_rate:.4f} "
                    f"Jaccard={val_metrics.jaccard:.4f} "
                    f"PRAUC={val_metrics.prauc:.4f} "
                    f"F1={val_metrics.f1:.4f} "
                    f"Avg Meds={val_metrics.num_meds:.4f} "
                )
                
                if val_metrics.jaccard > best_score:
                    best_score = val_metrics.jaccard
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    
                    logger.info(
                        f"New best model! Score={best_score:.4f} "
                        f"(Jaccard={val_metrics.jaccard:.4f}, "
                        f"DDI Rate={val_metrics.ddi_rate:.4f})"
                    )
                    
                    torch.save(
                        {
                            "model_state_dict": best_state,
                            "epoch": epoch,
                            "jaccard": float(best_score),
                            "val_metrics": val_metrics,
                            "model_cfg": model.cfg,
                            "train_cfg": self.config,
                        },
                        checkpoint_path
                    )
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
                    history["best_metrics"] = val_metrics
                    history["best_epoch"] = best_epoch
            
            total_training_time = time.time() - training_start_time
            history["training_time"] = total_training_time
            
            logger.info(f"Training completed in {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
            return history

    def _optimiser(self, params: Iterable[torch.nn.Parameter]):
        match self.config.optimiser:
            case "Adam":
                return optim.Adam(
                    params=params, 
                    lr=self.config.learning_rate, 
                    weight_decay=self.config.weight_decay
                )
            case "AdamW":
                return optim.AdamW(
                    params=params, 
                    lr=self.config.learning_rate, 
                    weight_decay=self.config.weight_decay
                )
            case "Adagrad":
                return optim.Adagrad(
                    params=params, 
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            case _:
                raise ValueError(f"Unknown optimiser: {self.config.optimiser}")
    
    def _scheduler(
        self, 
        optimiser: optim.Optimizer, 
        warmup_steps: int, 
        total_steps: int, 
        min_lr_ratio: float = 0.1
    ):
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps

            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

        return optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)