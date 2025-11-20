from src.core.interfaces.splitter import Splitter
from src.data.context import ProcessedDataContext


class GAMENETSplitter(Splitter):
    def split(self, context: ProcessedDataContext):
        data = context.records
        split_point = int(len(data) * 2 / 3)
        data_train = data[:split_point]
        eval_len = int(len(data[split_point:]) / 2)
        data_test = data[split_point:split_point + eval_len]
        data_val = data[split_point + eval_len:]
        return data_train, data_val, data_test
