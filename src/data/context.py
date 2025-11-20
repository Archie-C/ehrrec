from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class ProcessedDataContext:
    """
    Container for processed dataset artifacts (records, vocabularies,
    adjacency matrices, miscellaneous metadata) that different models
    can query without forcing `main.py` to pass each item explicitly.
    """

    name: str
    records: List[Any]
    vocab: Dict[str, Any]
    adjacency: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_adj(self, key: str, default=None):
        return self.adjacency.get(key, default)

    def vocab_sizes(self) -> Tuple[int, int, int]:
        diag_vocab = self.vocab.get("diagnoses_vocab")
        proc_vocab = self.vocab.get("procedures_vocab")
        med_vocab = self.vocab.get("medication_vocab")
        if not all([diag_vocab, proc_vocab, med_vocab]):
            raise KeyError("Missing vocab entries to compute sizes")
        return (
            len(diag_vocab.idx_to_word),
            len(proc_vocab.idx_to_word),
            len(med_vocab.idx_to_word),
        )
