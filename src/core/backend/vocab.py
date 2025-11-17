class Vocab(object):
    def __init__(self):
        self.idx_to_word = {}
        self.word_to_idx = {}
    
    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word_to_idx:
                self.idx_to_word[len(self.word_to_idx)] = word
                self.word_to_idx[word] = len(self.word_to_idx)
