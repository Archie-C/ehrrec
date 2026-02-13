from typing import Tuple
import numpy as np
import pandas as pd
from itertools import combinations

from src.preprocess.BasePreprocessor import AbstractPreprocessor, ProcessedSpec, ProcessedTables, RequiredTables

# TODO: Check the leakage properly
# Potentially filtering before splitting is leakage
class MIMIC3Preprocessor(AbstractPreprocessor):
    """
    Preprocessor for MIMIC3 dataset.
    """
    def __init__(self) -> None:
        super().__init__()
        self.top_n_diagnoses: int | None = None
        self.top_n_procedures: int | None = None
        self.top_n_prescriptions: int | None = None
        self.use_chrono_split: bool | None = None
    
    def requires(self) -> RequiredTables:
        return RequiredTables(
            tables={"diagnoses": pd.DataFrame, "procedures": pd.DataFrame, "prescriptions": pd.DataFrame},
            meta={},
            artefacts={"ndc_rxcui": dict, "rxcui_atc3": pd.DataFrame},
            schema_version="0.1.0"
        )
    
    def produces(self) -> ProcessedSpec:
        return ProcessedSpec(
            tables={"train": pd.DataFrame, "val": pd.DataFrame, "test": pd.DataFrame},
            meta={"columns": ["SUBJECT_ID", "HADM_ID", "DIAG_CODES", "MED_CODES", "PROC_CODES", "DIAG_LEN", "PROC_LEN", "MED_LEN", "ADMITTIME", "DIAG_IDS", "PROC_IDS", "MED_IDS"]},
            artefacts={"ddi_adj": "", "ehr_adj": ""},
            schema_version="0.1.0"
        )
    
    def process(self, loaded_data: RequiredTables) -> ProcessedTables:
        diagnoses = loaded_data.tables["diagnoses"]
        procedures = loaded_data.tables["procedures"]
        prescriptions = loaded_data.tables["prescriptions"]
        admissions = loaded_data.tables["admissions"]
        
        ndc_to_rxcui = loaded_data.artefacts["ndc_rxcui"]
        rxcui_to_atc3 = loaded_data.artefacts["rxcui_atc3"]
        ddi_raw = loaded_data.artefacts["ddi"]
        cid2atc3 = loaded_data.artefacts["cid2atc3"]
        
        diagnoses_processed = self._process_diagnoses(diagnoses, self.top_n_diagnoses)
        procedures_processed = self._process_procedures(procedures, self.top_n_procedures)
        prescriptions_processed = self._process_prescriptions(prescriptions, self.top_n_prescriptions)
        prescriptions_processed = self._convert_ndc_to_atc(prescriptions_processed, ndc_to_rxcui, rxcui_to_atc3)
        
        merged = self._merge(diagnoses_processed, procedures_processed, prescriptions_processed, admissions)
        train, val, test, info = self._split_per_patient_chronological(merged)
        
        vocab = self.build_id_maps(train)
        train, val, test = self.encode(train, val, test, vocab)
        
        ehr_adj, ddi_adj = self._create_adjacency_matrices(ddi_raw, cid2atc3, vocab["med"][1], vocab["med"][0], train, 40)
        
        return ProcessedTables(
            tables={"train": train, "val": val, "test": test},
            meta={"columns": ["SUBJECT_ID", "HADM_ID", "DIAG_CODES", "MED_CODES", "PROC_CODES", "DIAG_LEN", "PROC_LEN", "MED_LEN", "ADMITTIME", "DIAG_IDS", "PROC_IDS", "MED_IDS"]},
            artefacts={"ehr_adj": ehr_adj, "ddi_adj": ddi_adj, "vocab": vocab, "molecule": loaded_data.artefacts["molecule"]},
            schema_version="0.1.0"
        )
    
    # ----------------------------------------------------------------------
    #                               Diagnoses
    # ----------------------------------------------------------------------
    
    def _process_diagnoses(self, diagnoses: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
        df = diagnoses.copy()
        df = df.dropna(ignore_index=True)
        df = df.drop(columns=["SEQ_NUM", "ROW_ID"], errors="ignore")
        df = df.drop_duplicates(ignore_index=True)
        if top_n is not None:
            df = self._filter_diagnoses(df, top_n)
        df = df.sort_values(by=["SUBJECT_ID", "HADM_ID"], kind="stable")
        return df.reset_index(drop=True)

    def _filter_diagnoses(self, diagnoses: pd.DataFrame, top_n: int) -> pd.DataFrame:
        counts = (
            diagnoses.groupby("ICD9_CODE", sort=False)
            .size()
            .reset_index(name="COUNT")
            .sort_values("COUNT", ascending=False, kind="stable")
        )
        top_codes = counts["ICD9_CODE"].head(top_n)
        return diagnoses[diagnoses["ICD9_CODE"].isin(top_codes)].reset_index(drop=True)
    
    # ----------------------------------------------------------------------
    #                               Procedures
    # ----------------------------------------------------------------------
    
    def _process_procedures(self, procedures: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
        df = procedures.copy()
        df = df.dropna(ignore_index=True)
        df = df.drop(columns=["ROW_ID"], errors="ignore")
        df = df.drop_duplicates(ignore_index=True)
        if top_n is not None:
            df = self._filter_procedures(df, top_n)
        df = df.sort_values(by=["SUBJECT_ID", "HADM_ID"], kind="stable")
        return df.reset_index(drop=True)
    
    def _filter_procedures(self, procedures: pd.DataFrame, top_n: int) -> pd.DataFrame:
        counts = (
            procedures.groupby("ICD9_CODE", sort=False)
            .size()
            .reset_index(name="COUNT")
            .sort_values("COUNT", ascending=False, kind="stable")
        )
        top_codes = counts["ICD9_CODE"].head(top_n)
        return procedures[procedures["ICD9_CODE"].isin(top_codes)].reset_index(drop=True)
    
    # ----------------------------------------------------------------------
    #                               Prescriptions
    # ----------------------------------------------------------------------
    
    # This gives one medication code per hospital admission, chosen from the earliest ICU event
    # If the same drug was administered multiple times during the same admission, only keep one record
    def _process_prescriptions(self, prescriptions: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
        df = prescriptions.copy()
        
        drop_cols = [
            "ROW_ID", "DRUG_TYPE", "DRUG_NAME_POE", "DRUG_NAME_GENERIC",
            "FORMULARY_DRUG_CD", "GSN", "PROD_STRENGTH", "DOSE_VAL_RX",
            "DOSE_UNIT_RX", "FORM_VAL_DISP", "FORM_UNIT_DISP",
            "ROUTE", "ENDDATE", "DRUG"
        ]
        df = df.drop(columns=drop_cols, errors="ignore")

        
        if "NDC" in df.columns:
            df = df[df["NDC"].notna() & (df["NDC"] != "0")]

        
        if "STARTDATE" in df.columns:
            df["STARTDATE"] = pd.to_datetime(df["STARTDATE"], errors="coerce")
        if "ICUSTAY_ID" in df.columns:
            df["ICUSTAY_ID"] = pd.to_numeric(df["ICUSTAY_ID"], errors="coerce").astype("Int64")
        
        required = [c for c in ["SUBJECT_ID", "HADM_ID", "STARTDATE", "ICUSTAY_ID", "NDC"] if c in df.columns]
        df = df.dropna(subset=required).reset_index(drop=True)
        
        df = df.drop_duplicates(ignore_index=True)
        df = df.sort_values(by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE"], kind="stable")
        
        df = self._filter_first24h_med_set(df)
        
        df = df.drop(columns=["ICUSTAY_ID"], errors="ignore")
        df = df.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID", "NDC"], ignore_index=True)
        
        if top_n is not None:
            df = self._filter_prescriptions(df, top_n)
        
        hadm_counts = df.groupby("SUBJECT_ID", sort=False)["HADM_ID"].nunique()
        keep_subjects = hadm_counts[hadm_counts >= 2].index
        df = df[df["SUBJECT_ID"].isin(keep_subjects)].reset_index(drop=True)
        
        return df
    
    def _filter_first24h_med_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keeps meds in first 24 hours of each ICU stay, then drops repeats of the same med
        within that ICU stay (set behaviour).
        """
        t0 = df.groupby(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], sort=False)["STARTDATE"].transform("min")
        df = df[df["STARTDATE"] <= (t0 + pd.Timedelta(hours=24))]

        # Set behaviour within ICU stay: keep unique meds (drop repeated administrations)
        df = df.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "NDC"], ignore_index=True)

        return df
    
    def _filter_prescriptions(self, prescriptions: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """
        Top-N by *admission document frequency* (recommended):
        count how many admissions contain each med at least once.
        """
        adm = prescriptions.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID", "NDC"])
        counts = (
            adm.groupby("NDC", sort=False)
            .size()
            .reset_index(name="COUNT")
            .sort_values("COUNT", ascending=False, kind="stable")
        )
        top_codes = counts["NDC"].head(top_n)
        return prescriptions[prescriptions["NDC"].isin(top_codes)].reset_index(drop=True)
    
    # ----------------------------------------------------------------------
    #                        Convert NDC to ADC
    # ----------------------------------------------------------------------
    # this converts NDCs to ATC3 codes for drugs
    def _convert_ndc_to_atc(self, prescriptions: pd.DataFrame, ndc_to_rxcui: dict, rxcui_to_at3: pd.DataFrame) -> pd.DataFrame:
        df = prescriptions.copy()
        df["NDC"] = df["NDC"].astype(str)
        
        ndc_raw = df["NDC"].astype(str)
        ndc_norm = (
            ndc_raw.str.replace(r"\.0$", "", regex=True)
                .str.replace(r"\D", "", regex=True)
                .str.zfill(11)
        )
        df["RXCUI"] = ndc_norm.map(ndc_to_rxcui)
        
        df["RXCUI"] = pd.to_numeric(df["RXCUI"], errors="coerce")    
        df = df.dropna(subset=["RXCUI"]).copy()
        df["RXCUI"] = df["RXCUI"].astype("int64")
        
        map_df = rxcui_to_at3.copy()
        
        if {"YEAR", "MONTH"}.issubset(map_df.columns):
            map_df = map_df.sort_values(["RXCUI", "YEAR", "MONTH"], kind="stable")
            map_df = map_df.drop_duplicates(subset=["RXCUI"], keep="last")
        else:
            map_df = map_df.drop_duplicates(subset=["RXCUI"])
        
        keep_cols = [c for c in ["RXCUI", "ATC4"] if c in map_df.columns]
        map_df = map_df[keep_cols].copy()
        
        map_df["RXCUI"] = pd.to_numeric(map_df["RXCUI"], errors="coerce").astype("Int64")
        map_df = map_df.dropna(subset=["RXCUI"])
        map_df["RXCUI"] = map_df["RXCUI"].astype("int64")
        
        df = df.merge(map_df, on="RXCUI", how="inner")
        
        df["ATC3"] = df["ATC4"].astype(str).str.slice(0, 4)
        
        df = df.drop(columns=["NDC", "RXCUI", "ATC4"], errors="ignore")
        
        df = df.drop_duplicates(ignore_index=True)

        return df
    
    # ----------------------------------------------------------------------
    #                               Merge
    # ----------------------------------------------------------------------
    
    def _merge(self, diag: pd.DataFrame, proc: pd.DataFrame, meds: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
        diag = diag.copy()
        proc = proc.copy()
        meds = meds.copy()
        
        #meds = meds.drop(columns=["STARTDATE"], errors="ignore")
        
        keys = (
            diag[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
            .merge(proc[["SUBJECT_ID", "HADM_ID"]].drop_duplicates(), on=["SUBJECT_ID", "HADM_ID"], how="inner")
            .merge(meds[["SUBJECT_ID", "HADM_ID"]].drop_duplicates(),  on=["SUBJECT_ID", "HADM_ID"], how="inner")
        )
        
        def adm_set(df: pd.DataFrame, code_col: str, out_col: str) -> pd.DataFrame:
            tmp = df.merge(keys, on=["SUBJECT_ID", "HADM_ID"], how="inner")
            tmp = tmp.dropna(subset=[code_col])
            tmp = tmp.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID", code_col])
            return (
                tmp.groupby(["SUBJECT_ID", "HADM_ID"], sort=False)[code_col]
                .agg(lambda s: sorted(s.astype(str).tolist()))
                .reset_index()
                .rename(columns={code_col: out_col})
            )
        diag_adm = adm_set(diag, "ICD9_CODE", "DIAG_CODES")
        proc_adm = adm_set(proc, "ICD9_CODE", "PROC_CODES")
        med_adm  = adm_set(meds,  "ATC3",      "MED_CODES")
        
        merged = (
            diag_adm.merge(med_adm, on=["SUBJECT_ID", "HADM_ID"], how="inner")
                    .merge(proc_adm, on=["SUBJECT_ID", "HADM_ID"], how="inner")
                    .reset_index(drop=True)
        )
        
        merged["DIAG_LEN"] = merged["DIAG_CODES"].map(len)
        merged["PROC_LEN"] = merged["PROC_CODES"].map(len)
        merged["MED_LEN"]  = merged["MED_CODES"].map(len)
        
        admissions_df = admissions[["SUBJECT_ID", "HADM_ID", "ADMITTIME"]].copy()
        admissions_df["ADMITTIME"] = pd.to_datetime(admissions_df["ADMITTIME"], errors="coerce")
        
        merged = merged.merge(admissions_df, on=["SUBJECT_ID", "HADM_ID"], how="left")

        return merged
    
    # ----------------------------------------------------------------------
    #                               Splitting
    # ----------------------------------------------------------------------
    
    # Within each patient
    def _split_per_patient_chronological(
        self, 
        visits: pd.DataFrame,
        *, 
        time_col: str = "ADMITTIME",
        patient_col: str = "SUBJECT_ID",
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), 
        use_patient_time: str = "last",
        min_visits_per_patient: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Patient-disjoint + time split with quantile cutoffs to target ratios (default 80/10/10).

        We compute a per-patient timestamp (last or first admission time), then:
        TRAIN: patient_time <  q80
        VAL:   q80 <= patient_time < q90
        TEST:  patient_time >= q90

        Returns: (train_df, val_df, test_df, info_dict)
        """
        r_train, r_val, r_test = ratios
        if abs((r_train + r_val + r_test) - 1.0) > 1e-9:
            raise ValueError("ratios must sum to 1.0")
        if use_patient_time not in {"last", "first"}:
            raise ValueError("use_patient_time must be 'last' or 'first'")

        df = visits.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df[df[time_col].notna()].copy()

        if min_visits_per_patient > 1:
            sizes = df.groupby(patient_col, sort=False).size()
            keep = sizes[sizes >= min_visits_per_patient].index
            df = df[df[patient_col].isin(keep)].copy()

        agg = df.groupby(patient_col)[time_col]
        patient_time = (agg.max() if use_patient_time == "last" else agg.min()).rename("patient_time")
        patient_time = patient_time.sort_values()

        # Quantile cut points based on PATIENTS (not visits)
        q_val = patient_time.quantile(r_train, interpolation="linear")          # 80%
        q_test = patient_time.quantile(r_train + r_val, interpolation="linear") # 90%

        train_pat = set(patient_time[patient_time < q_val].index)
        val_pat = set(patient_time[(patient_time >= q_val) & (patient_time < q_test)].index)
        test_pat = set(patient_time[patient_time >= q_test].index)

        # Disjoint sanity
        assert train_pat.isdisjoint(val_pat)
        assert train_pat.isdisjoint(test_pat)
        assert val_pat.isdisjoint(test_pat)

        train = df[df[patient_col].isin(train_pat)].reset_index(drop=True)
        val = df[df[patient_col].isin(val_pat)].reset_index(drop=True)
        test = df[df[patient_col].isin(test_pat)].reset_index(drop=True)

        info = {
            "ratios": ratios,
            "use_patient_time": use_patient_time,
            "min_visits_per_patient": min_visits_per_patient,
            "cutoffs": {
                "val_start_inclusive_q80": str(pd.Timestamp(q_val)),
                "test_start_inclusive_q90": str(pd.Timestamp(q_test)),
            },
            "patients": {
                "total": int(patient_time.shape[0]),
                "train": int(len(train_pat)),
                "val": int(len(val_pat)),
                "test": int(len(test_pat)),
            },
            "visits": {
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
            },
        }
        return train, val, test, info
        
    # ----------------------------------------------------------------------
    #                               Vocab
    # ----------------------------------------------------------------------
    
    def build_id_maps(self, df: pd.DataFrame):
        def build_map(series, special_tokens=("<PAD>", "<UNK>")):
            codes = sorted({c for lst in series.dropna() for c in lst})
            code_list = list(special_tokens) + codes
            code2idx = {c: i for i, c in enumerate(code_list)}
            idx2code = {i: c for i, c in enumerate(code_list)}
            return code2idx, idx2code
        
        diag2idx, idx2diag = build_map(df["DIAG_CODES"])
        proc2idx, idx2proc = build_map(df["PROC_CODES"])
        med2idx,  idx2med  = build_map(df["MED_CODES"])

        return {
            "diag": (diag2idx, idx2diag),
            "proc": (proc2idx, idx2proc),
            "med":  (med2idx,  idx2med),
        }
    
    def encode_split(self, df, vocab):
        diag2idx, _ = vocab["diag"]
        proc2idx, _ = vocab["proc"]
        med2idx,  _ = vocab["med"]

        def encode_list(lst, m):
            unk = m["<UNK>"]
            # set semantics + deterministic
            return sorted({m.get(x, unk) for x in lst})

        out = df.copy()
        out["DIAG_IDS"] = out["DIAG_CODES"].map(lambda xs: encode_list(xs, diag2idx))
        out["PROC_IDS"] = out["PROC_CODES"].map(lambda xs: encode_list(xs, proc2idx))
        out["MED_IDS"]  = out["MED_CODES"].map(lambda xs: encode_list(xs, med2idx))
        return out
    
    def encode(self, train, val, test, vocab):
        train_e = self.encode_split(train, vocab)
        val_e   = self.encode_split(val, vocab)
        test_e  = self.encode_split(test, vocab)
        return train_e, val_e, test_e
    
    # ----------------------------------------------------------------------
    #                         Adjacency matrices
    # ----------------------------------------------------------------------
    
    def _create_adjacency_matrices(self, ddi_raw: pd.DataFrame, cid2atc3, medidx2code, medcode2idx, train: pd.DataFrame, topk: int ):
        med_voc_size = len(medidx2code)
        med_tokens = list(medidx2code)
        
        vocab_atc3 = set()
        for tok in med_tokens:
            if not isinstance(tok, str):
                continue
            if tok.startswith("<"):
                continue
            if len(tok) < 4:
                continue
            vocab_atc3.add(tok[:4])
        

        ehr_adj = np.zeros((med_voc_size, med_voc_size), dtype=np.uint8)
        
        for _, item in train.iterrows():
            med_set = item["MED_IDS"]
            meds = sorted(set(med_set))
            for i, j in combinations(meds, 2):
                if i == j:
                    continue
                ehr_adj[i, j] = 1
                ehr_adj[j, i] = 1
        
        # Filters by the 40 least frequent side effects
        se_counts = (
            ddi_raw.groupby("Side Effect Name", sort=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=True, kind="stable")
        )
        
        top_se = se_counts.head(topk)[["Side Effect Name"]]
        ddi_pairs = (
            ddi_raw.merge(top_se, on="Side Effect Name", how="inner")[["STITCH 1", "STITCH 2"]]
            .drop_duplicates()
        )
        
        ddi_adj = np.zeros((med_voc_size, med_voc_size), dtype=np.uint8)
        for _, row in ddi_pairs.iterrows():
            cid1, cid2 = row["STITCH 1"], row["STITCH 2"]
            for atc3_i in cid2atc3.get(cid1, set()):
                for atc3_j in cid2atc3.get(cid2, set()):
                    if atc3_i == atc3_j:
                        continue
                    if atc3_i in medcode2idx and atc3_j in medcode2idx:
                        idx_i = medcode2idx[atc3_i]
                        idx_j = medcode2idx[atc3_j]
                        ddi_adj[idx_i, idx_j] = 1
                        ddi_adj[idx_j, idx_i] = 1

        return ehr_adj, ddi_adj