
from src.models.BaseModel import BaseModel
import time
import numpy as np
from src.utils.logging import get_logger
from src.utils.metrics import evaluate_multilabel_sets

logger = get_logger("Most Popular Baseline")

class MostPopular(BaseModel):
    def __init__(self, vocab_size, ddi_adj, k=15):
        super().__init__()
        self.most_pop_counts = None
        self.vocab_size = vocab_size
        self.ddi_adj = ddi_adj
        self.k = k
        
    def fit(self, train, val):
        start_train_time = time.time()
        if self.most_pop_counts is None:
            self.most_pop_counts = self._create_most_pop_counts(train)
        
        preds = []
        targets = []
        
        for item in val:
            target = item[-1]
            pred = self._predict_from_most_pop().tolist()
            preds.append(pred)
            targets.append(target)
            
        metrics = evaluate_multilabel_sets(targets, preds, self.ddi_adj, ignore_ids=[0, 1])
            
        training_time = time.time() - start_train_time
        
        logger.info("Train and validation complete")
        logger.info(f"Time taken {training_time:.4f} seconds ({(training_time / 60):.4f} mins)")
        logger.info(f"Val metrics: Jaccard: {metrics['jaccard']:.4f} F1: {metrics['f1']:.4f} Recall: {metrics['recall']:.4f} Precision: {metrics['precision']:.4f} DDI_pred: {metrics['ddi_rate_pred']:.4f} DDI_true: {metrics['ddi_rate_true']:.4f}")
        
        return metrics

    def _create_most_pop_counts(self, train):
        counts = np.zeros(self.vocab_size[2])
        for item in train:
            meds = item[-1]
            counts[meds] += 1
        return counts

    def _predict_from_most_pop(self):
        preds = np.argsort(self.most_pop_counts)[-self.k:][::-1]
        return preds

    def predict(self, data):
        if self.most_pop_counts is None:
            logger.error("model.fit() must be called before predict.")
            
        start_time = time.time()
            
        preds = []
        targets = []
        
        for item in data:
            target = item[-1]
            pred = self._predict_from_most_pop().tolist()
            preds.append(pred)
            targets.append(target)
            
        metrics = evaluate_multilabel_sets(targets, preds, self.ddi_adj, ignore_ids=[0, 1])
            
        t =  time.time() - start_time
        
        logger.info("Predictions complete")
        logger.info(f"Time taken {t:.4f} seconds ({(t / 60):.4f} mins)")
        logger.info(f"Val metrics: Jaccard: {metrics['jaccard']:.4f} F1: {metrics['f1']:.4f} Recall: {metrics['recall']:.4f} Precision: {metrics['precision']:.4f} DDI_pred: {metrics['ddi_rate_pred']:.4f} DDI_true: {metrics['ddi_rate_true']:.4f}")
        
        return metrics