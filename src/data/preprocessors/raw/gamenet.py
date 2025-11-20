from pathlib import Path
from src.core.backend.vocab import Vocab
from src.core.interfaces.preprocessor import Preprocessor
from src.core.interfaces.table import Table

import os
import dill
from collections import defaultdict
import numpy as np
import logging
import pandas as pd 

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None


class GAMENetPreprocessor(Preprocessor):
    def __init__(self, output_dir: Path, filenames: dict[str, str] | None = None, log_level: int = logging.INFO, show_progress: bool = True):
        self.output_dir = output_dir
        self.filenames = filenames or {
            "records": "records.pkl",
            "vocab": "vocab.pkl",
            "filtered_data": "filtered_data.pkl",
            "ehr_adj": "ehr_adj.pkl",
            "ddi_adj": "ddi_adj.pkl"
        }

        # logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.propagate = False

        self.show_progress = show_progress and (tqdm is not None)
        if show_progress and tqdm is None:
            self.logger.warning("show_progress=True but tqdm not available; progress bars disabled.")

    # ----------------------------------------------------------------------
    #                               MAIN RUN
    # ----------------------------------------------------------------------
    def run(self, data: dict):
        self.logger.info("Starting GAMENet preprocessing")

        procedures = self._process_procedures(data["procedures"])
        diagnoses  = self._process_diagnoses(data["diagnoses"])
        diagnoses  = self._filter_diagnoses(diagnoses, 2000)

        meds = self._process_medications(data["medications"])
        meds = self._convert_ndc_to_atc(
            med_pd=meds,
            rxnorm2atc=data["rxnorm_to_atc"],
            ndc_to_rxnorm=data["ndc_to_rxnorm"]
        )

        merged = self._merge(diagnoses, procedures, meds)

        diag_vocab, pro_vocab, med_vocab = self._create_vocab(merged)
        records = self._create_records(merged, diag_vocab, pro_vocab, med_vocab)

        ehr_adj, ddi_adj = self._create_adjacency_matrices(
            data["ddi"],
            data["cid_to_atc"],
            med_vocab,
            records,
            topk=40
        )

        self.logger.info("Completed preprocessing.")
        self.print_statistics(merged)

        return {
            "filtered_data": merged,
            "vocab": {
                "diagnoses_vocab": diag_vocab,
                "procedures_vocab": pro_vocab,
                "medication_vocab": med_vocab
            },
            "records": records,
            "ehr_adj": ehr_adj,
            "ddi_adj": ddi_adj,
        }

    # ----------------------------------------------------------------------
    #                        PROCESSING FUNCTIONS
    # ----------------------------------------------------------------------
    def _process_procedures(self, data: pd.DataFrame) -> pd.DataFrame:
        data.drop(columns=['ROW_ID'], inplace=True)
        data.drop_duplicates(inplace=True)
        data.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
        data.drop(columns=['SEQ_NUM'], inplace=True)
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    def _process_medications(self, med_pd: pd.DataFrame) -> pd.DataFrame:
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
        
        def filter_first24hour_med(med_pd: pd.DataFrame) -> pd.DataFrame:
            med_pd_new = med_pd.drop(columns=['NDC'])
            med_pd_new = med_pd_new.groupby(by=['SUBJECT_ID','HADM_ID','ICUSTAY_ID']).head(1).reset_index(drop=True)
            med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','STARTDATE'])
            med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
            return med_pd_new
        med_pd = filter_first24hour_med(med_pd)
        
        med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
        med_pd = med_pd.drop_duplicates()
        med_pd = med_pd.reset_index(drop=True)
        
        # visit > 2
        def process_visit_lg2(med_pd: pd.DataFrame) -> pd.DataFrame:
            a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
            a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
            a = a[a['HADM_ID_Len'] > 1]
            return a 
        med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)    
        med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')    
        
        return med_pd.reset_index(drop=True)

    def _process_diagnoses(self, diag_pd: pd.DataFrame) -> pd.DataFrame:
        diag_pd.dropna(inplace=True)
        diag_pd.drop(columns=['SEQ_NUM','ROW_ID'],inplace=True)
        diag_pd.drop_duplicates(inplace=True)
        diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
        return diag_pd.reset_index(drop=True)


    # ----------------------------------------------------------------------
    #               EXACT MATCH: top-2000 diagnoses filter
    # ----------------------------------------------------------------------
    def _filter_diagnoses(self, diag_pd: pd.DataFrame, amount: int) -> pd.DataFrame:
        diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:amount-1, 'ICD9_CODE'])]
        
        return diag_pd.reset_index(drop=True)

    # ----------------------------------------------------------------------
    #        EXACT MATCH: NDC → RXCUI → ATC4 mapping (no behavioural drift)
    # ----------------------------------------------------------------------
    def _convert_ndc_to_atc(self, med_pd: pd.DataFrame, rxnorm2atc: pd.DataFrame, ndc_to_rxnorm) -> pd.DataFrame:
        with open(ndc_to_rxnorm, 'r') as f:
            ndc2rxnorm = eval(f.read())
        med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
        med_pd.dropna(inplace=True)

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

    # ----------------------------------------------------------------------
    #                             MERGING
    # ----------------------------------------------------------------------
    def _merge(self, diag_pd: pd.DataFrame, pro_pd: pd.DataFrame, med_pd: pd.DataFrame) -> pd.DataFrame:
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

    def _create_vocab(self, data: Table):
        diagnoses_vocab = Vocab()
        procedures_vocab = Vocab()
        medication_vocab = Vocab()

        row_iter = data.iterrows()
        if self.show_progress:
            row_iter = tqdm(row_iter, desc="Creating vocabs", unit="rows")
        count = 0
        for _, row in row_iter:
            diagnoses_vocab.add_sentence(row['ICD9_CODE'])
            procedures_vocab.add_sentence(row['PRO_CODE'])
            medication_vocab.add_sentence(row['NDC'])
            count += 1
            if count % 1000 == 0:
                self.logger.debug("Vocab building: processed %d rows", count)

        self.logger.debug("Created vocabs (processed %d rows)", count)
        return diagnoses_vocab, procedures_vocab, medication_vocab

    def _create_records(self, data: Table, diagnoses_vocab: Vocab, procedures_vocab: Vocab, medication_vocab: Vocab) -> list:
        records = []
        subject_ids = list(data['SUBJECT_ID'].unique())
        iterator = subject_ids
        if self.show_progress:
            iterator = tqdm(subject_ids, desc="Creating records", unit="patients")
        for idx, subject_id in enumerate(iterator):
            item_df = data[data['SUBJECT_ID'] == subject_id]
            patient = []
            for _, row in item_df.iterrows():
                admission = []
                admission.append([diagnoses_vocab.word_to_idx[i] for i in row['ICD9_CODE']])
                admission.append([procedures_vocab.word_to_idx[i] for i in row['PRO_CODE']])
                admission.append([medication_vocab.word_to_idx[i] for i in row['NDC']])
                patient.append(admission)
            records.append(patient)
            if (idx + 1) % 100 == 0:
                self.logger.debug("Created records for %d patients so far", idx + 1)
        self.logger.debug("Created records array for %d patients", len(records))
        return records

    def _create_adjacency_matrices(self, ddi_file_path: str, cid_atc_path: str, med_voc: Vocab, records: list, topk=40):
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
    
    def print_statistics(self, df: pd.DataFrame):
        print("===== DATASET STATISTICS =====")


        # Basic counts
        patients = df["SUBJECT_ID"].nunique()
        events = len(df)

        print(f"Patients:                 {patients:,}")
        print(f"Clinical events:          {events:,}")

        # Flatten list columns
        unique_diag = set().union(*df["ICD9_CODE"])
        unique_med  = set().union(*df["NDC"])
        unique_pro  = set().union(*df["PRO_CODE"])

        print(f"Unique diagnoses:         {len(unique_diag):,}")
        print(f"Unique medicines:         {len(unique_med):,}")
        print(f"Unique procedures:        {len(unique_pro):,}")

        # Group by patient
        g = df.groupby("SUBJECT_ID")

        # Per patient: total visits
        visit_counts = g.size()

        # Per patient: unique counts
        diag_counts = g["ICD9_CODE"].apply(lambda x: len(set().union(*x)))
        med_counts  = g["NDC"].apply(lambda x: len(set().union(*x)))
        pro_counts  = g["PRO_CODE"].apply(lambda x: len(set().union(*x)))

        print("\n----- Per-Patient Averages -----")
        print(f"Average diagnoses:         {diag_counts.mean():.4f}")
        print(f"Average medicines:         {med_counts.mean():.4f}")
        print(f"Average procedures:        {pro_counts.mean():.4f}")
        print(f"Average visits:            {visit_counts.mean():.4f}")

        print("\n----- Maximum Per-Patient Values -----")
        print(f"Max diagnoses:             {diag_counts.max()}")
        print(f"Max medicines:             {med_counts.max()}")
        print(f"Max procedures:            {pro_counts.max()}")
        print(f"Max visits:                {visit_counts.max()}")

        print("===== END =====")

    def save(self, data: dict):
        """
        Save processed outputs using dill.
        One vocab file containing:
            {
                "diagnoses_vocab": ...,
                "procedures_vocab": ...,
                "medication_vocab": ...
            }
        """

        os.makedirs(self.output_dir, exist_ok=True)

        # output files
        paths = {
            "filtered_data": "filtered_data.pkl",
            "vocab":         "vocab.pkl",         # ← single file
            "records":       "records.pkl",
            "ehr_adj":       "ehr_adj.pkl",
            "ddi_adj":       "ddi_adj.pkl",
        }

        # filtered + merged DataFrame/Table
        with open(os.path.join(self.output_dir, paths["filtered_data"]), "wb") as f:
            dill.dump(data["filtered_data"], f)

        # vocab in one file
        vocab_dict = {
            "diagnoses_vocab":  data["vocab"]["diagnoses_vocab"],
            "procedures_vocab": data["vocab"]["procedures_vocab"],
            "medication_vocab": data["vocab"]["medication_vocab"],
        }
        with open(os.path.join(self.output_dir, paths["vocab"]), "wb") as f:
            dill.dump(vocab_dict, f)

        # records
        with open(os.path.join(self.output_dir, paths["records"]), "wb") as f:
            dill.dump(data["records"], f)

        # adjacency matrices
        with open(os.path.join(self.output_dir, paths["ehr_adj"]), "wb") as f:
            dill.dump(data["ehr_adj"], f)

        with open(os.path.join(self.output_dir, paths["ddi_adj"]), "wb") as f:
            dill.dump(data["ddi_adj"], f)
