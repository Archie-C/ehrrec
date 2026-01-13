import pandas as pd
import dill

from src.data.raw import create_raw_loader
from src.data.preprocessors.raw.gamenet import GAMENetPreprocessor
from _helpers import assert_df_equal, assert_list_equal, assert_array_equal

from omegaconf import OmegaConf

failures = []
def record_fail(name, error):
    failures.append((name, error))
    print(f"[FAIL] {name}: {error}")
# --------------------------------------------------------
# LOAD OLD PIPELINE OUTPUT
# --------------------------------------------------------
old_data = pd.read_pickle("C:/Users/cararc/Documents/Code/GAMENet/data/data_final.pkl")
with open("C:/Users/cararc/Documents/Code/GAMENet/data/voc_final.pkl", "rb") as f:
    voc_old = dill.load(f)
diag_voc_old = voc_old["diag_voc"]
med_voc_old = voc_old["med_voc"]
pro_voc_old = voc_old["pro_voc"]

records_old = dill.load(open("C:/Users/cararc/Documents/Code/GAMENet/data/records_final.pkl", "rb"))
ehr_adj_old = dill.load(open("C:/Users/cararc/Documents/Code/GAMENet/data/ehr_adj_final.pkl", "rb"))
ddi_adj_old = dill.load(open("C:/Users/cararc/Documents/Code/GAMENet/data/ddi_A_final.pkl", "rb"))

dataset_config = {
    "paths": {
        "diagnoses": "data/raw/MIMIC-III/DIAGNOSES_ICD.csv",
        "procedures": "data/raw/MIMIC-III/PROCEDURES_ICD.csv",
        "medications": "data/raw/MIMIC-III/PRESCRIPTIONS.csv",
        "ndc_to_rxnorm": "data/raw/MIMIC-III/ndc2rxnorm_mapping.txt",
        "rxnorm_to_atc": "data/raw/MIMIC-III/ndc2atc_level4.csv",
        "cid_to_atc": "data/raw/MIMIC-III/drug-atc.csv",
        "ddi": "data/raw/MIMIC-III/drug-DDI.csv",
    },
    "kwargs": {
        "diagnoses": {"low_memory": False},
        "procedures": {"low_memory": False},
        "medications": {"low_memory": False},
        "ndc_to_rxnorm": {},
        "rxnorm_to_atc": {},
        "cid_to_atc": {},
        "ddi": {"low_memory": False},
    },
}
# --------------------------------------------------------
# RUN NEW PIPELINE
# --------------------------------------------------------
raw_loader = create_raw_loader(OmegaConf.create({"name": "mimic3"}))
raw_inputs = raw_loader.load(OmegaConf.create(dataset_config).paths, OmegaConf.create(dataset_config).kwargs)
pre = GAMENetPreprocessor(output_dir="test_out/")
new_output = pre.run(raw_inputs)

new_data = new_output["filtered_data"].df
new_diag_vocab = new_output["vocab"]["diagnoses_vocab"]
new_med_vocab  = new_output["vocab"]["medication_vocab"]
new_pro_vocab  = new_output["vocab"]["procedures_vocab"]

new_records = new_output["records"]
new_ehr_adj = new_output["ehr_adj"]
new_ddi_adj = new_output["ddi_adj"]


# --------------------------------------------------------
# TEST 1 — Final merged dataset identical
# --------------------------------------------------------
try:
    assert_df_equal(old_data, new_data, "Merged Dataset")
except AssertionError as e:
    record_fail("Merged Dataset", e)


# --------------------------------------------------------
# TEST 2 — Vocabulary exact match
# --------------------------------------------------------
try:
    assert_list_equal(diag_voc_old.idx2word, new_diag_vocab.idx_to_word, "Diagnosis Vocab")
except AssertionError as e:
    record_fail("Diagnosis Vocab", e)

try:
    assert_list_equal(med_voc_old.idx2word, new_med_vocab.idx_to_word, "Medication Vocab")
except AssertionError as e:
    record_fail("Medication Vocab", e)

try:
    assert_list_equal(pro_voc_old.idx2word, new_pro_vocab.idx_to_word, "Procedure Vocab")
except AssertionError as e:
    record_fail("Procedure Vocab", e)


# --------------------------------------------------------
# TEST 3 — Records identical
# --------------------------------------------------------
try:
    assert_list_equal(records_old, new_records, "Patient Records")
except AssertionError as e:
    record_fail("Patient Records", e)


# --------------------------------------------------------
# TEST 4 — Adjacency matrices identical
# --------------------------------------------------------
try:
    assert_array_equal(ehr_adj_old, new_ehr_adj, "EHR Adj Matrix")
except AssertionError as e:
    record_fail("EHR Adj Matrix", e)

try:
    assert_array_equal(ddi_adj_old, new_ddi_adj, "DDI Adj Matrix")
except AssertionError as e:
    record_fail("DDI Adj Matrix", e)

print("\n======================")
if failures:
    print("TESTS COMPLETED WITH FAILURES\n")
    for name, error in failures:
        print(f" - {name}: FAILED")
else:
    print("ALL TESTS PASSED ✓")
print("======================\n")
