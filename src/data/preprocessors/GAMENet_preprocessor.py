from pathlib import Path
from src.core.backend.vocab import Vocab
from src.core.interfaces.preprocessor import Preprocessor
from src.core.interfaces.table import Table

import os
import dill
from collections import defaultdict
import numpy as np
import logging

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

        # optional progress bars with tqdm
        self.show_progress = show_progress and (tqdm is not None)
        if show_progress and tqdm is None:
            self.logger.warning("show_progress=True but tqdm not available; progress bars disabled.")

    def run(self, data: dict[str, Table]):
        self.logger.info("Starting GAMENet preprocessing")
        self.logger.debug("Input tables: %s", list(data.keys()))

        procedures = self._process_procedures(data["procedures"])
        self.logger.info("Processed procedures with shape %s", getattr(procedures, "shape", str(len(procedures))))

        diagnoses = self._process_diagnoses(data["diagnoses"])
        diagnoses = self._filter_diagnoses(diagnoses, 2000)
        self.logger.info("Processed & filtered diagnoses with shape %s", getattr(diagnoses, "shape", str(len(diagnoses))))

        medication = self._process_medications(data["medications"])
        medication = self._convert_ndc_to_atc(medication=medication, rxnorm_to_atc=data["rxnorm_to_atc"], ndc_to_rxnorm=data["ndc_to_rxnorm"])
        self.logger.info("Processed medication with shape %s", getattr(medication, "shape", str(len(medication))))

        processed_data = self._merge(diagnoses=diagnoses, procedures=procedures, medication=medication)
        self.logger.info("Merged data shape: %s", getattr(processed_data, "shape", str(len(processed_data))))

        diagnoses_vocab, procedures_vocab, medication_vocab = self._create_vocab(processed_data)
        self.logger.info("Created vocabularies: diagnoses(%d), procedures(%d), medication(%d)", 
                        len(diagnoses_vocab.idx_to_word), len(procedures_vocab.idx_to_word), len(medication_vocab.idx_to_word))

        records = self._create_records(processed_data, diagnoses_vocab, procedures_vocab, medication_vocab)
        self.logger.info("Created records for %d patients", len(records))

        ehr_adj, ddi_adj = self._create_adjacency_matrices(data["ddi"], data["cid_to_atc"], medication_vocab, records, 40)
        self.logger.info("Created adjacency matrices: ehr_adj(%s), ddi_adj(%s)", getattr(ehr_adj, "shape", ehr_adj.size if hasattr(ehr_adj, "size") else "unknown"), getattr(ddi_adj, "shape", ddi_adj.size if hasattr(ddi_adj, "size") else "unknown"))

        self.logger.info("GAMENet preprocessing completed")
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
        self.logger.info("Saving processed artifacts to %s", self.output_dir)
        data["filtered_data"].to_pickle(self.output_dir + self.filenames["filtered_data"])
        with open(self.output_dir + self.filenames["vocab"], "wb") as f:
            dill.dump(obj=data["vocab"], file=f)
        with open(self.output_dir + self.filenames["records"], "wb") as f:
            dill.dump(obj=data["records"], file=f)
        with open(self.output_dir + self.filenames["ehr_adj"], "wb") as f:
            dill.dump(obj=data["ehr_adj"], file=f)
        with open(self.output_dir + self.filenames["ddi_adj"], "wb") as f:
            dill.dump(obj=data["ddi_adj"], file=f)
        self.logger.info("Successfully saved all files")

    def _process_procedures(self, data: Table) -> Table:
        data = data.as_type({"ICD9_CODE": "category"})
        data = data.drop(columns=["ROW_ID"])
        data = data.drop_duplicates()
        data = data.sort_values(by=["SUBJECT_ID", "HADM_ID", "SEQ_NUM"])
        data = data.drop(columns=["SEQ_NUM"])
        data = data.drop_duplicates()
        data = data.reset_index(drop=True)
        self.logger.debug("Procedures: post-processed rows=%s", getattr(data, "shape", len(data)))
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
        subject_ids_to_keep = a.filter(lambda row: row['HADM_ID_Len'] > 1).select(['SUBJECT_ID'])
        data = data.merge(subject_ids_to_keep, on="SUBJECT_ID", how="inner")
        data = data.reset_index(drop=True)
        self.logger.debug("Medications: post-processed rows=%s", getattr(data, "shape", len(data)))
        return data

    def _process_diagnoses(self, data: Table) -> Table:
        data = data.drop_na()
        data = data.drop(columns=['SEQ_NUM', 'ROW_ID'])
        data = data.drop_duplicates()
        data = data.sort_values(by=['SUBJECT_ID', "HADM_ID"])
        data = data.reset_index(drop=True)
        self.logger.debug("Diagnoses: post-processed rows=%s", getattr(data, "shape", len(data)))
        return data

    def _filter_diagnoses(self, diagnoses: Table, amount: int) -> Table:
        diagnoses_count: Table = diagnoses.groupby('ICD9_CODE').size().reset_index().rename({"size": 'count'}).sort_values(by='count', ascending=False).reset_index(drop=True)
        top_codes: Table = diagnoses_count.select(columns=['ICD9_CODE']).head(amount)
        diagnoses = diagnoses.filter(lambda row: row['ICD9_CODE'] in top_codes['ICD9_CODE'].to_list())
        self.logger.debug("Filtered diagnoses to top %s codes; remaining rows=%s", amount, getattr(diagnoses, "shape", len(diagnoses)))
        return diagnoses

    def _convert_ndc_to_atc(self, medication: Table, rxnorm_to_atc: Table, ndc_to_rxnorm) -> Table:
        medication = medication.assign(NDC=lambda t: t['NDC'].astype(str).str.replace(r'\.0$', '', regex=True))
        medication = medication.assign(RXCUI=lambda t: t['NDC'].map(ndc_to_rxnorm))
        try:
            sample = getattr(medication, "head", lambda n=2: None)(2)
            self.logger.debug("Medication sample after RXCUI mapping: %s", sample)
        except Exception:
            self.logger.debug("Medication sample unavailable")
        medication = medication.drop_na()

        rxnorm_to_atc = rxnorm_to_atc.drop(columns=['YEAR','MONTH','NDC'])
        rxnorm_to_atc = rxnorm_to_atc.drop_duplicates(subset=['RXCUI'])

        medication = medication.filter(lambda row: row['RXCUI'] != '')
        try:
            sample2 = getattr(medication, "head", lambda n=2: None)(2)
            self.logger.debug("Medication sample after filtering empty RXCUI: %s", sample2)
        except Exception:
            self.logger.debug("Medication sample 2 unavailable")
        medication = medication.as_type({"RXCUI": "int64"})
        medication = medication.reset_index(drop=True)
        medication = medication.merge(right=rxnorm_to_atc, on=["RXCUI"])
        medication = medication.drop(columns=['NDC', 'RXCUI'])
        medication = medication.rename(mapping={"ATC4": "NDC"})
        medication = medication.assign(NDC=lambda t: t['NDC'].map(lambda x: x[:4]))
        medication = medication.drop_duplicates()
        medication = medication.reset_index(drop=True)
        self.logger.debug("Medication: converted to ATC4 (NDC col shortened); rows=%s", getattr(medication, "shape", len(medication)))
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
        self.logger.debug("Merged data rows=%s", getattr(data, "shape", len(data)))
        return data

    def _create_vocab(self, data: Table):
        diagnoses_vocab = Vocab()
        procedures_vocab = Vocab()
        medication_vocab = Vocab()

        row_iter = data.row_iter()
        if self.show_progress:
            row_iter = tqdm(row_iter, desc="Creating vocabs", unit="rows")
        count = 0
        for row in row_iter:
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
            item_df: Table = data.filter(lambda row: row['SUBJECT_ID'] == subject_id)
            patient = []
            for row in item_df.row_iter():
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

    def _create_adjacency_matrices(self, ddi: Table, cid_to_atc: str, medication_vocab: Vocab, records: list, topk=40):
        cid_to_atc_dic = defaultdict(set)
        medication_vocab_size = len(medication_vocab.idx_to_word)
        medication_unique_word = [medication_vocab.idx_to_word[i] for i in range(medication_vocab_size)]
        atc3_atc4_dic = defaultdict(set)

        if self.show_progress:
            med_iter = tqdm(medication_unique_word, desc="Indexing ATC groups", unit="meds")
        else:
            med_iter = medication_unique_word
        for item in med_iter:
            atc3_atc4_dic[item[:4]].add(item)
        self.logger.debug("Indexed ATC level-3 -> level-4 map; groups=%d", len(atc3_atc4_dic))

        with open(cid_to_atc, 'r') as f:
            lines = f if not self.show_progress else tqdm(f, desc="Reading cid->atc", unit="lines")
            for line in lines:
                line_ls = line[:-1].split(',')
                cid = line_ls[0]
                atcs = line_ls[1:]
                for atc in atcs:
                    if len(atc3_atc4_dic[atc[:4]]) != 0:
                        cid_to_atc_dic[cid].add(atc[:4])
        self.logger.debug("Loaded cid->atc mappings; mapped cids=%d", len(cid_to_atc_dic))

        ddi_most: Table = ddi.groupby(['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename({"size": "count"}).sort_values(by='count', ascending=False).reset_index(drop=True)
        ddi_most = ddi_most.slice_rows(-topk, None)

        filter_ddi: Table = ddi.merge(right=ddi_most.select(columns=['Side Effect Name']), on=['Side Effect Name'], how='inner')
        ddi = filter_ddi.select(columns=["STITCH 1", "STITCH 2"]).drop_duplicates().reset_index(drop=True)
        self.logger.debug("Filtered DDI to top %d side-effects; rows=%s", topk, getattr(ddi, "shape", len(ddi)))

        ehr_adj = np.zeros((medication_vocab_size, medication_vocab_size))
        records_iter = records if not self.show_progress else tqdm(records, desc="Building EHR adjacency", unit="patients")
        for patient in records_iter:
            for adm in patient:
                med_set = adm[2]
                for i, med_i in enumerate(med_set):
                    for j, med_j in enumerate(med_set):
                        if j<=i:
                            continue
                        ehr_adj[med_i, med_j] = 1
                        ehr_adj[med_j, med_i] = 1
        self.logger.debug("Built EHR adjacency matrix; nonzero count=%d", int(ehr_adj.sum()))

        ddi_adj = np.zeros((medication_vocab_size, medication_vocab_size))
        ddi_rows = ddi.row_iter()
        if self.show_progress:
            ddi_rows = tqdm(list(ddi_rows), desc="Building DDI adjacency", unit="pairs")
        for row in ddi_rows:
            cid1 = row['STITCH 1']
            cid2 = row['STITCH 2']
            for atc_i in cid_to_atc_dic.get(cid1, []):
                for atc_j in cid_to_atc_dic.get(cid2, []):
                    for i in atc3_atc4_dic.get(atc_i, []):
                        for j in atc3_atc4_dic.get(atc_j, []):
                            if medication_vocab.word2idx.get(i) != medication_vocab.word2idx.get(j):
                                ii = medication_vocab.word2idx[i]
                                jj = medication_vocab.word2idx[j]
                                ddi_adj[ii, jj] = 1
                                ddi_adj[jj, ii] = 1
        self.logger.debug("Built DDI adjacency matrix; nonzero count=%d", int(ddi_adj.sum()))

        return ehr_adj, ddi_adj
    def _create_adjacency_matrices(self, ddi: Table, cid_to_atc: str, medication_vocab: Vocab, records: list, topk=40):
        cid_to_atc_dic = defaultdict(set)
        medication_vocab_size = len(medication_vocab.idx_to_word)
        medication_unique_word = [medication_vocab.idx_to_word[i] for i in range(medication_vocab_size)]
        atc3_atc4_dic = defaultdict(set)

        if self.show_progress:
            med_iter = tqdm(medication_unique_word, desc="Indexing ATC groups", unit="meds")
        else:
            med_iter = medication_unique_word
        for item in med_iter:
            atc3_atc4_dic[item[:4]].add(item)
        self.logger.debug("Indexed ATC level-3 -> level-4 map; groups=%d", len(atc3_atc4_dic))

        with open(cid_to_atc, 'r') as f:
            lines = f if not self.show_progress else tqdm(f, desc="Reading cid->atc", unit="lines")
            for line in lines:
                line_ls = line[:-1].split(',')
                cid = line_ls[0]
                atcs = line_ls[1:]
                for atc in atcs:
                    if len(atc3_atc4_dic[atc[:4]]) != 0:
                        cid_to_atc_dic[cid].add(atc[:4])
        self.logger.debug("Loaded cid->atc mappings; mapped cids=%d", len(cid_to_atc_dic))

        ddi_most: Table = ddi.groupby(['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename({"size": "count"}).sort_values(by='count', ascending=False).reset_index(drop=True)
        ddi_most = ddi_most.slice_rows(-topk, None)

        filter_ddi: Table = ddi.merge(right=ddi_most.select(columns=['Side Effect Name']), on=['Side Effect Name'], how='inner')
        ddi = filter_ddi.select(columns=["STITCH 1", "STITCH 2"]).drop_duplicates().reset_index(drop=True)
        self.logger.debug("Filtered DDI to top %d side-effects; rows=%s", topk, getattr(ddi, "shape", len(ddi)))

        ehr_adj = np.zeros((medication_vocab_size, medication_vocab_size))
        records_iter = records if not self.show_progress else tqdm(records, desc="Building EHR adjacency", unit="patients")
        for patient in records_iter:
            for adm in patient:
                med_set = adm[2]
                for i, med_i in enumerate(med_set):
                    for j, med_j in enumerate(med_set):
                        if j<=i:
                            continue
                        ehr_adj[med_i, med_j] = 1
                        ehr_adj[med_j, med_i] = 1
        self.logger.debug("Built EHR adjacency matrix; nonzero count=%d", int(ehr_adj.sum()))

        ddi_adj = np.zeros((medication_vocab_size, medication_vocab_size))
        ddi_rows = ddi.row_iter()
        if self.show_progress:
            ddi_rows = tqdm(list(ddi_rows), desc="Building DDI adjacency", unit="pairs")
        for row in ddi_rows:
            cid1 = row['STITCH 1']
            cid2 = row['STITCH 2']
            for atc_i in cid_to_atc_dic.get(cid1, []):
                for atc_j in cid_to_atc_dic.get(cid2, []):
                    for i in atc3_atc4_dic.get(atc_i, []):
                        for j in atc3_atc4_dic.get(atc_j, []):
                            if medication_vocab.word_to_idx.get(i) != medication_vocab.word_to_idx.get(j):
                                ii = medication_vocab.word_to_idx[i]
                                jj = medication_vocab.word_to_idx[j]
                                ddi_adj[ii, jj] = 1
                                ddi_adj[jj, ii] = 1
        self.logger.debug("Built DDI adjacency matrix; nonzero count=%d", int(ddi_adj.sum()))

        return ehr_adj, ddi_adj
    