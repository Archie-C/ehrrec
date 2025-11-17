from pathlib import Path
from src.core.backend.vocab import Vocab
from src.core.interfaces.preprocessor import Preprocessor
from src.core.interfaces.table import Table

import dill
from collections import defaultdict
import numpy as np

class GAMENetPreprocessor(Preprocessor):
    def __init__(self, output_dir: Path, filenames: dict[str, str] | None = None):
        self.output_dir = output_dir
        self.filenames = filenames or {
            "records": "records.pkl",
            "vocab": "vocab.pkl",
            "filtered_data": "filtered_data.pkl",
            "ehr_adj": "ehr_adj.pkl",
            "ddi_adj": "ddi_adj.pkl"
        }
        
    def run(self, data: dict[str, Table]):
        procedures = self._process_procedures(data["procedures"])
        
        diagnoses = self._process_diagnoses(data["diagnoses"])
        diagnoses = self._filter_diagnoses(diagnoses, 2000)
        
        medication = self._process_medications(data["medications"])
        medication = self._convert_ndc_to_atc(medication=medication, rxnorm_to_atc=data["rxnorm_to_atc"], ndc_to_rxnorm=data["ndc_to_rxnorm"])
        
        processed_data = self._merge(diagnoses=diagnoses, procedures=procedures, medication=medication)
        
        diagnoses_vocab, procedures_vocab, medication_vocab = self._create_vocab(processed_data)
        
        records = self._create_records(processed_data, diagnoses_vocab, procedures_vocab, medication_vocab)
        
        ehr_adj, ddi_adj = self._create_adjacency_matrices(data["ddi"], data["cid_to_atc"], medication_vocab, records, 40)
        return {
            "filtered_data": processed_data,
            "vocab": {
                "diagnoses_vocab": diagnoses_vocab,
                "procedures_vocab": procedures_vocab,
                "medication_vocab": medication_vocab
            },
            "records": records,
            "ehr_adj": ehr_adj,
            "ddi_adj": ddi_adj,
        }
    
    def save(self, data: dict):
        data["filtered_data"].to_pickle(self.output_dir + self.filenames["filtered_data"])
        dill.dump(obj=data["vocab"], file=open(self.output_dir + self.filenames["vocab"], "wb"))
        dill.dump(obj=data["records"], file=open(self.output_dir + self.filenames["records"], "wb"))
        dill.dump(obj=data["ehr_adj"], file=open(self.output_dir + self.filenames["ehr_adj"]))
        dill.dump(obj=data["ddi_adj"], file=open(self.output_dir + self.filenames["ddi_adj"]))
        print("Successfully saved all files")
        
    def _process_procedures(self, data: Table) -> Table:
        data = data.as_type({"ICD9_CODE": "category"})
        data = data.drop(columns=["ROW_ID"])
        data = data.drop_duplicates()
        data = data.sort_values(by=["SUBJECT_ID", "HADM_ID", "SEQ_NUM"])
        data = data.drop(columns=["SEQ_NUM"])
        data = data.drop_duplicates()
        data = data.reset_index(drop=True)
        return data
    
    def _process_medications(self, data: Table) -> Table:
        data = data.as_type({"NDC": "category"})
        data = data.drop(columns=[
                'ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                'FORMULARY_DRUG_CD','GSN','PROD_STRENGTH','DOSE_VAL_RX',
                'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP','FORM_UNIT_DISP',
                'ROUTE','ENDDATE','DRUG'],
                axis=1
            )
        data = data.filter(lambda row: row['NDC'] != '0')
        data = data.fillna(method="pad")
        data = data.drop_na()
        data = data.drop_duplicates()
        data = data.as_type({"ICUSTAY_ID": "int64"})
        data = data.to_datetime(columns=["STARTDATE"], format='%Y-%m-%d %H:%M:%S')
        data = data.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'])
        data = data.reset_index(drop=True)
        
        # Filtering by the first 24 hours after admission
        data_new: Table = data.select(['SUBJECT_ID','HADM_ID','ICUSTAY_ID','STARTDATE'])
        data_new = data_new.groupby(['SUBJECT_ID','HADM_ID','ICUSTAY_ID']).head(1).reset_index(drop=True)
        data = data.merge(right=data_new, on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'])
        data = data.drop(columns=['STARTDATE', 'ICUSTAY_ID'])
        data = data.drop_duplicates()
        data = data.reset_index(drop=True)
        
        a = data.select(['SUBJECT_ID', 'HADM_ID']).groupby('SUBJECT_ID').agg({'HADM_ID': 'unique'}).reset_index()
        a = a.assign(HADM_ID_Len=lambda t: t['HADM_ID'].map(len))
        a = a.filter(lambda row: row['HADM_ID_Len'] > 1)
        data = a.reset_index(drop=True)
        
        data = data.merge(right=data.select(['SUBJECT_ID']), on="SUBJECT_ID", how='inner')
        data = data.reset_index(drop=True)
        return data
    
    def _process_diagnoses(self, data: Table) -> Table:
        data = data.drop_na()
        data = data.drop(columns=['SEQ_NUM', 'ROW_ID'])
        data = data.drop_duplicates()
        data = data.sort_values(by=['SUBJECT_ID', "HADM_ID"])
        data = data.reset_index(drop=True)
        return data
    
    def _filter_diagnoses(self, diagnoses: Table, amount: int) -> Table:
        diagnoses_count: Table = diagnoses.groupby('ICD9_CODE').size().reset_index().rename({"size": 'count'}).sort_values(by='count', ascending=False).reset_index(drop=True)
        top_codes: Table = diagnoses_count.select(columns=['ICD9_CODE']).head(amount)
        diagnoses = diagnoses.filter(lambda row: row['ICD9_CODE'] in top_codes['ICD9_CODE'].to_list())
        return diagnoses
        
    def _convert_ndc_to_atc(self, medication: Table, rxnorm_to_atc: Table, ndc_to_rxnorm) -> Table:
        medication = medication.assign(RXCUI=lambda t: t['NDC'].map(ndc_to_rxnorm))
        medication = medication.drop_na()
        
        rxnorm_to_atc = rxnorm_to_atc.drop(columns=['YEAR','MONTH','NDC'])
        rxnorm_to_atc = rxnorm_to_atc.drop_duplicates(subset=['RXCUI'])
        
        medication = medication.filter(lambda row: row['RXCUI'] != '')
        medication = medication.as_type({"RXCUI": "int64"})
        medication = medication.reset_index(drop=True)
        medication = medication.merge(right=rxnorm_to_atc, on=["RXCUI"])
        medication = medication.drop(columns=['NDC', 'RXCUI'])
        medication = medication.rename(mapping={"ATC4": "NDC"})
        medication = medication.assign(NDC=lambda t: t['NDC'].map(lambda x: x[:4]))
        medication = medication.drop_duplicates()
        medication = medication.reset_index(drop=True)
        return medication
    
    def _merge(self, diagnoses: Table, procedures: Table, medication: Table) -> Table:
        diagnoses_key: Table = diagnoses.select(columns=["SUBJECT_ID", "HADM_ID"]).drop_duplicates()
        procedures_key: Table = procedures.select(columns=["SUBJECT_ID", "HADM_ID"]).drop_duplicates()
        medication_key: Table = medication.select(columns=["SUBJECT_ID", "HADM_ID"]).drop_duplicates()
        
        combined_key: Table = medication_key.merge(right=diagnoses_key, on=["SUBJECT_ID", "HADM_ID"], how='inner')
        combined_key = combined_key.merge(right=procedures_key, on=["SUBJECT_ID", "HADM_ID"], how='inner')
        
        diagnoses = diagnoses.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        procedures = procedures.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        medication = medication.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
        
        diagnoses = diagnoses.groupby(["SUBJECT_ID", "HADM_ID"]).agg({'ICD9_CODE': 'unique'}).reset_index()
        procedures = procedures.groupby(["SUBJECT_ID", "HADM_ID"]).agg({'ICD9_CODE': 'unique'}).reset_index().rename({"ICD9_CODE": "PRO_CODE"})
        medication = medication.groupby(["SUBJECT_ID", "HADM_ID"]).agg({'NDC': 'unique'}).reset_index()
        
        procedures = procedures.assign(PRO_CODE=lambda t: t['PRO_CODE'].map(lambda x: list(x)))
        medication = medication.assign(NDC=lambda t: t['NDC'].map(lambda x: list(x)))
        
        data = diagnoses.merge(right=medication, on=["SUBJECT_ID", "HADM_ID"], how='inner')
        data = data.merge(right=procedures, on=["SUBJECT_ID", "HADM_ID"], how='inner')
        data = data.assign(NDC_Len=lambda t: t['NDC'].map(lambda x: len(x)))
        return data
    
    def _create_vocab(self, data: Table):
        diagnoses_vocab = Vocab()
        procedures_vocab = Vocab()
        medication_vocab = Vocab()
        
        for row in data.row_iter():
            diagnoses_vocab.add_sentence(row['ICD9_CODE'])
            procedures_vocab.add_sentence(row['PRO_CODE'])
            medication_vocab.add_sentence(row['NDC'])
    
        return diagnoses_vocab, procedures_vocab, medication_vocab
    
    def _create_records(self, data: Table, diagnoses_vocab: Vocab, procedures_vocab: Vocab, medication_vocab: Vocab) -> list:
        records = []
        for subject_id in data['SUBJECT_ID'].unique():
            item_df: Table = data.filter(lambda row: row['SUBJECT_ID'] == subject_id)
            patient = []
            for row in item_df.row_iter():
                admission = []
                admission.append([diagnoses_vocab.word2idx[i] for i in row['ICD9_CODE']])
                admission.append([procedures_vocab.word2idx[i] for i in row['PRO_CODE']])
                admission.append([medication_vocab.word2idx[i] for i in row['NDC']])
                patient.append(admission)
            records.append(patient) 
        return records
    
    def _create_adjacency_matrices(ddi: Table, cid_to_atc: str, medication_vocab: Vocab, records: list, topk=40):
        cid_to_atc_dic = defaultdict(set)
        medication_vocab_size = len(medication_vocab.idx_to_word)
        medication_unique_word = [medication_vocab.idx_to_word[i] for i in range(medication_vocab_size)]
        atc3_atc4_dic = defaultdict(set)
        
        for item in medication_unique_word:
            atc3_atc4_dic[item[:4]].add(item)
        
        with open(cid_to_atc, 'r') as f:
            for line in f:
                line_ls = line[:-1].split(',')
                cid = line_ls[0]
                atcs = line_ls[1:]
                for atc in atcs:
                    if len(atc3_atc4_dic[atc[:4]]) != 0:
                        cid_to_atc_dic[cid].add(atc[:4])
                        
        ddi_most: Table = ddi.groupby(['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename({"size": "count"}).sort_values(by='count', ascending=False).reset_index(drop=True)
        ddi_most = ddi_most.slice_rows(-topk, None)
        
        filter_ddi: Table = ddi.merge(right=ddi_most.select(columns=['Side Effect Name']), on=['Side Effect Name'], how='inner')
        ddi = filter_ddi.select(columns=["STITCH 1", "STITCH 2"]).drop_duplicates().reset_index(drop=True)
        
        ehr_adj = np.zeros((medication_vocab_size, medication_vocab_size))
        for patient in records:
            for adm in patient:
                med_set = adm[2]
                for i, med_i in enumerate(med_set):
                    for j, med_j in enumerate(med_set):
                        if j<=i:
                            continue
                        ehr_adj[med_i, med_j] = 1
                        ehr_adj[med_j, med_i] = 1
                        
        ddi_adj = np.zeros((medication_vocab_size, medication_vocab_size))                
        for row in ddi.row_iter():
            # ddi
            cid1 = row['STITCH 1']
            cid2 = row['STITCH 2']
            
            # cid -> atc_level3
            for atc_i in cid_to_atc_dic[cid1]:
                for atc_j in cid_to_atc_dic[cid2]:
                    
                    # atc_level3 -> atc_level4
                    for i in atc3_atc4_dic[atc_i]:
                        for j in atc3_atc4_dic[atc_j]:
                            if medication_vocab.word2idx[i] != medication_vocab.word2idx[j]:
                                ddi_adj[medication_vocab.word2idx[i], medication_vocab.word2idx[j]] = 1
                                ddi_adj[medication_vocab.word2idx[j], medication_vocab.word2idx[i]] = 1
        
        return ehr_adj, ddi_adj
    