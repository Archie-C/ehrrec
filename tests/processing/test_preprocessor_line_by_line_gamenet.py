import pandas as pd
import dill
import numpy as np

from src.core.backend.vocab import Vocab
from src.data.raw import create_raw_loader
from src.data.preprocessors.raw.gamenet import GAMENetPreprocessor
from _helpers import assert_df_equal, assert_records_equal, assert_vocab_equal

from omegaconf import OmegaConf

diag_file = "C:/Users/cararc/Documents/Code/ehrrec/data/raw/MIMIC-III/DIAGNOSES_ICD.csv"
procedure_file = "C:/Users/cararc/Documents/Code/ehrrec/data/raw/MIMIC-III/PROCEDURES_ICD.csv"
med_file = "C:/Users/cararc/Documents/Code/ehrrec/data/raw/MIMIC-III/PRESCRIPTIONS.csv"
ndc2rxnorm_file = "C:/Users/cararc/Documents/Code/ehrrec/data/raw/MIMIC-III/ndc2rxnorm_mapping.txt"
ndc2atc_file = "C:/Users/cararc/Documents/Code/ehrrec/data/raw/MIMIC-III/ndc2atc_level4.csv"
ddi = "C:/Users/cararc/Documents/Code/ehrrec/data/raw/MIMIC-III/drug-DDI.csv"
cid_to_atc_file = "C:/Users/cararc/Documents/Code/ehrrec/data/raw/MIMIC-III/drug-atc.csv"

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
        # EXACT old behaviour: ICD9_CODE loaded as category
        "diagnoses":  {
            "low_memory": False,
        },

        # EXACT old behaviour: ICD9_CODE loaded as category
        "procedures": {
            "low_memory": False,
            "dtype": {"ICD9_CODE": "category"},
        },

        # EXACT old behaviour: NDC loaded as category
        "medications": {
            "low_memory": False,
            "dtype": {"NDC": "category"},
        },

        "ndc_to_rxnorm": {},
        "rxnorm_to_atc": {},
        "cid_to_atc": {},
        "ddi": {"low_memory": False},
    },
}


raw_loader = create_raw_loader(OmegaConf.create({"name": "mimic3"}))
raw_inputs = raw_loader.load(OmegaConf.create(dataset_config).paths, OmegaConf.create(dataset_config).kwargs)

def process_diag():
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM','ROW_ID'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
    return diag_pd.reset_index(drop=True)

def process_procedure():
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE':'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)
    
    return pro_pd

def process_med():
    med_pd = pd.read_csv(med_file, dtype={'NDC':'category'})
    # filter
    med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                     'FORMULARY_DRUG_CD','GSN','PROD_STRENGTH','DOSE_VAL_RX',
                     'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP','FORM_UNIT_DISP',
                      'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
    med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')    
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)
    
    def filter_first24hour_med(med_pd):
        med_pd_new = med_pd.drop(columns=['NDC'])
        med_pd_new = med_pd_new.groupby(by=['SUBJECT_ID','HADM_ID','ICUSTAY_ID']).head(1).reset_index(drop=True)
        med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','STARTDATE'])
        med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
        return med_pd_new
    med_pd = filter_first24hour_med(med_pd)
#     med_pd = med_pd.drop(columns=['STARTDATE'])
    
    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    
    # visit > 2
    def process_visit_lg2(med_pd):
        a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
        a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
        a = a[a['HADM_ID_Len'] > 1]
        return a 
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)    
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')    
    
    return med_pd.reset_index(drop=True)

def test_process_diagnoses():
    old_diag = process_diag()  # from the old script
    new_diag = GAMENetPreprocessor(output_dir='test')._process_diagnoses(raw_inputs["diagnoses"])
    assert_df_equal(old_diag, new_diag, "Process Diagnoses ")
    
def test_process_procedures():
    old_pro = process_procedure()

    new_pro = GAMENetPreprocessor(output_dir='test')._process_procedures(raw_inputs["procedures"])

    assert_df_equal(old_pro.astype({"ICD9_CODE": "string"}),
                new_pro.astype({"ICD9_CODE": "string"}),
                "Process Procedures")

def test_process_medications():
    old_data = process_med()  # from the old script
    new_data = GAMENetPreprocessor(output_dir='test')._process_medications(raw_inputs["medications"])
    assert_df_equal(old_data.astype({"NDC": "string"}), new_data.astype({"NDC": "string"}), "Process Medication ")


def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]
    
    return diag_pd.reset_index(drop=True)

def test_diag_filter():
    pre = GAMENetPreprocessor(output_dir='test')
    new_diag = pre._process_diagnoses(raw_inputs["diagnoses"])
    old_data = filter_2000_most_diag(new_diag)
    new_data = pre._filter_diagnoses(new_diag, 2000)
    
    assert_df_equal(old_data, new_data, "Filter Diagnoses ")

def ndc2atc4(med_pd):
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR','MONTH','NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index = med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)
    
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4':'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()    
    med_pd = med_pd.reset_index(drop=True)
    return med_pd

def test_convert_ndc_to_atc():
    pre = GAMENetPreprocessor(output_dir='test')
    medication = pre._process_medications(raw_inputs["medications"])
    old_data = ndc2atc4(medication)
    new_data = pre._convert_ndc_to_atc(medication, rxnorm2atc=raw_inputs["rxnorm_to_atc"],ndc_to_rxnorm=raw_inputs["ndc_to_rxnorm"])
    assert_df_equal(old_data, new_data, "Conversion from NDC to ATC ")

def merge(diag_pd, pro_pd, med_pd):
    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    
    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    
    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()  
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})  
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    return data

def test_merge():
    pre = GAMENetPreprocessor(output_dir='test')
    new_diag = pre._process_diagnoses(raw_inputs["diagnoses"])
    new_diag = pre._filter_diagnoses(new_diag, 2000)
    new_proc = pre._process_procedures(raw_inputs["procedures"])
    medication = pre._process_medications(raw_inputs["medications"])
    medication = pre._convert_ndc_to_atc(medication, rxnorm2atc=raw_inputs["rxnorm_to_atc"],ndc_to_rxnorm=raw_inputs["ndc_to_rxnorm"])
    old_data = merge(new_diag, new_proc, medication)
    
    new_data = pre._merge(new_diag, new_proc, medication)
    assert_df_equal(old_data, new_data, "Merge")

                
def create_str_token_mapping(df):
    diag_voc = Vocab()
    med_voc = Vocab()
    pro_voc = Vocab()
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['NDC'])
        pro_voc.add_sentence(row['PRO_CODE'])
    
    return diag_voc, med_voc, pro_voc
    
def test_vocab():
    pre = GAMENetPreprocessor(output_dir='test')
    new_diag = pre._process_diagnoses(raw_inputs["diagnoses"])
    new_diag = pre._filter_diagnoses(new_diag, 2000)
    new_proc = pre._process_procedures(raw_inputs["procedures"])
    medication = pre._process_medications(raw_inputs["medications"])
    medication = pre._convert_ndc_to_atc(medication, rxnorm2atc=raw_inputs["rxnorm_to_atc"],ndc_to_rxnorm=raw_inputs["ndc_to_rxnorm"])
    merge = pre._merge(new_diag, new_proc, medication)
    new_diag_vocab, new_pro_vocab, new_med_vocab = pre._create_vocab(merge)
    old_diag_vocab, old_med_vocab, old_pro_vocab = create_str_token_mapping(merge)
    
    assert_vocab_equal(old_diag_vocab, new_diag_vocab, "Diagnosis Vocab")
    assert_vocab_equal(old_pro_vocab,  new_pro_vocab,  "Procedure Vocab")
    assert_vocab_equal(old_med_vocab,  new_med_vocab,  "Medication Vocab")


def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word_to_idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word_to_idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word_to_idx[i] for i in row['NDC']])
            patient.append(admission)
        records.append(patient) 
    return records


def test_records():
    pre = GAMENetPreprocessor(output_dir='test')
    new_diag = pre._process_diagnoses(raw_inputs["diagnoses"])
    new_diag = pre._filter_diagnoses(new_diag, 2000)
    new_proc = pre._process_procedures(raw_inputs["procedures"])
    medication = pre._process_medications(raw_inputs["medications"])
    medication = pre._convert_ndc_to_atc(medication, rxnorm2atc=raw_inputs["rxnorm_to_atc"],ndc_to_rxnorm=raw_inputs["ndc_to_rxnorm"])
    merge = pre._merge(new_diag, new_proc, medication)
    new_diag_vocab, new_pro_vocab, new_med_vocab = pre._create_vocab(merge)

    old_data = create_patient_record(merge, new_diag_vocab, new_med_vocab, new_pro_vocab)
    new_data = pre._create_records(merge, new_diag_vocab, new_pro_vocab, new_med_vocab)
    
    assert_records_equal(old_data, new_data, "Records ")
    
def build_adj_matrices(records, med_voc, cid_atc_path, ddi_file_path, topk=40):
    """
    Unit test version of the adjacency-matrix builder.

    Parameters
    ----------
    records : list
        Preprocessed patient records (list of admissions).
    med_voc : object
        Vocabulary object with fields:
            - idx2word: list[str]
            - word2idx: dict[str, int]
    cid_atc_path : str
        Path to the drug-atc.csv file.
    ddi_file_path : str
        Path to the drug-DDI.csv file.
    topk : int
        Number of most common DDI categories to keep.
    """

    import pandas as pd
    import numpy as np
    from collections import defaultdict

    # -------- vocab prep --------
    med_voc_size = len(med_voc.idx_to_word)
    med_unique_word = [med_voc.idx_to_word[i] for i in range(med_voc_size)]

    # atc3 → list(atc4 codes)
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)

    # -------- CID → ATC3 mapping --------
    cid2atc_dic = defaultdict(set)
    with open(cid_atc_path, 'r') as f:
        for line in f:
            line_ls = line.strip().split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])

    # -------- DDI filtering --------
    ddi_df = pd.read_csv(ddi_file_path)

    ddi_most_pd = (
        ddi_df
        .groupby(['Polypharmacy Side Effect', 'Side Effect Name'])
        .size()
        .reset_index()
        .rename(columns={0: 'count'})
        .sort_values(by='count', ascending=False)
        .reset_index(drop=True)
    )

    ddi_most_pd = ddi_most_pd.iloc[-topk:, :]
    fliter_ddi_df = ddi_df.merge(
        ddi_most_pd[['Side Effect Name']],
        how='inner',
        on='Side Effect Name'
    )

    ddi_df = fliter_ddi_df[['STITCH 1', 'STITCH 2']].drop_duplicates()

    # -------- EHR adjacency --------
    ehr_adj = np.zeros((med_voc_size, med_voc_size))

    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j <= i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1

    # -------- DDI adjacency --------
    ddi_adj = np.zeros((med_voc_size, med_voc_size))

    for _, row in ddi_df.iterrows():
        cid1 = row['STITCH 1']
        cid2 = row['STITCH 2']

        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        idx_i = med_voc.word_to_idx[i]
                        idx_j = med_voc.word_to_idx[j]
                        if idx_i != idx_j:
                            ddi_adj[idx_i, idx_j] = 1
                            ddi_adj[idx_j, idx_i] = 1

    # -------- Assertions for testing --------

    # return matrices for further use
    return ehr_adj, ddi_adj

    
def test_matrices():
    pre = GAMENetPreprocessor(output_dir='test')
    new_diag = pre._process_diagnoses(raw_inputs["diagnoses"])
    new_diag = pre._filter_diagnoses(new_diag, 2000)
    new_proc = pre._process_procedures(raw_inputs["procedures"])
    medication = pre._process_medications(raw_inputs["medications"])
    medication = pre._convert_ndc_to_atc(medication, rxnorm2atc=raw_inputs["rxnorm_to_atc"],ndc_to_rxnorm=raw_inputs["ndc_to_rxnorm"])
    merge = pre._merge(new_diag, new_proc, medication)
    new_diag_vocab, new_pro_vocab, new_med_vocab = pre._create_vocab(merge)
    records = pre._create_records(merge, new_diag_vocab, new_pro_vocab, new_med_vocab)
    ehr_adj_new, ddi_adj_new = pre._create_adjacency_matrices(raw_inputs["ddi"], raw_inputs["cid_to_atc"], new_med_vocab, records, topk=40)
    ehr_adj_old, ddi_adj_old = build_adj_matrices(records, new_med_vocab, cid_atc_path=cid_to_atc_file, ddi_file_path=ddi, topk=40)
    
    diff = ddi_adj_new != ddi_adj_old

    print("Number of differing cells:", diff.sum())
    print("Indices where they differ:")
    print(np.argwhere(diff))

    # Also inspect mismatched values
    mismatch_indices = np.argwhere(diff)
    for (i, j) in mismatch_indices[:20]:   # show first 20 only
        print(
            f"({i},{j}) new={ddi_adj_new[i,j]}, old={ddi_adj_old[i,j]}"
        )
    
    assert ehr_adj_new.shape == ehr_adj_old.shape
    assert ddi_adj_new.shape == ddi_adj_old.shape

    assert (ehr_adj_new == ehr_adj_old).all()
    assert (ddi_adj_new == ddi_adj_old).all()
    
