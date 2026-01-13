from src.core.interfaces.loader import Loader


class TextLoader(Loader):
    def load(self, path: str):
        with open(path, 'r') as f:
            data = eval(f.read())
        return data