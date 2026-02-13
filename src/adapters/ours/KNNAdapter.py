from src.adapters.BaseAdapter import BaseAdapter
from src.preprocess.BasePreprocessor import ProcessedTables
from src.utils.one_hot_encode import make_xy_dense

class KNNAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
    
    def adapt(self, data: ProcessedTables):
        train = data.tables["train"]
        val = data.tables["val"]
        test = data.tables["test"]
        
        vocab = data.artefacts["vocab"]
        
        diag_vocab_size = len(vocab["diag"][1])
        proc_vocab_size = len(vocab["proc"][1])
        med_vocab_size  = len(vocab["med"][1])
        
        X_train, Y_train = make_xy_dense(train, diag_vocab_size, proc_vocab_size, med_vocab_size)
        X_val,   _   = make_xy_dense(val,   diag_vocab_size, proc_vocab_size, med_vocab_size)
        X_test,  _  = make_xy_dense(test,  diag_vocab_size, proc_vocab_size, med_vocab_size)
        return X_train, Y_train, X_val, val["MED_IDS"].tolist(), X_test, test["MED_IDS"].tolist()