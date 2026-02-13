from __future__ import annotations

from typing import Any, Dict,  Optional, Tuple

import numpy as np
import pandas as pd


def _as_df(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, dict):
        return pd.DataFrame(x)
    raise TypeError(f"Expected pandas DataFrame (or dict), got: {type(x)}")


def _safe_len(x: Any) -> float:
    """Return length for list/tuple/set/np arrays; NaN if not countable."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return float(len(x))
    # sometimes stored as comma-separated string
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return 0.0
        # if it looks like "1,2,3"
        if "," in s:
            return float(len([p for p in s.split(",") if p.strip()]))
        return 1.0
    # scalar code/id
    return 1.0


def _q(series: pd.Series, qs=(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {f"p{int(q*100):02d}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"p{int(q*100):02d}"] = float(np.quantile(s.to_numpy(), q))
    return out


def _fmt_int(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def _fmt_float(x: Any, ndp: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{float(x):,.{ndp}f}"
    except Exception:
        return str(x)


def _print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _detect_id_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    subj = "SUBJECT_ID" if "SUBJECT_ID" in df.columns else None
    hadm = "HADM_ID" if "HADM_ID" in df.columns else None
    return subj, hadm


def _detect_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("ADMITTIME", "ADMIT_TIME", "ADMISSION_TIME", "INTIME"):
        if c in df.columns:
            return c
    return None


def _detect_len_col(df: pd.DataFrame, prefix: str) -> Optional[str]:
    # your meta includes DIAG_LEN / PROC_LEN / MED_LEN
    for c in (f"{prefix}_LEN", f"{prefix}LEN", f"{prefix}_COUNT"):
        if c in df.columns:
            return c
    return None


def _detect_codes_or_ids_col(df: pd.DataFrame, prefix: str) -> Optional[str]:
    # prefer *_IDS if present; else *_CODES
    for c in (f"{prefix}_IDS", f"{prefix}_CODES"):
        if c in df.columns:
            return c
    return None


def mimic3_dataset_stats(
    processed: Any,
    *,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    max_unique_examples: int = 3,
) -> Dict[str, Any]:
    """
    Given a ProcessedTables-like object:
      - processed.tables: dict with 'train'/'val'/'test' -> DataFrame
      - processed.artefacts: may contain 'vocab', 'ehr_adj', 'ddi_adj'
      - processed.meta: may contain 'columns'
    Prints nicely formatted stats and returns them as a dict.

    Assumes MIMIC-III style columns where available:
      SUBJECT_ID, HADM_ID, DIAG_IDS/DIAG_CODES, PROC_IDS/PROC_CODES, MED_IDS/MED_CODES,
      DIAG_LEN, PROC_LEN, MED_LEN, ADMITTIME
    """
    # grab tables
    tables = getattr(processed, "tables", None)
    if tables is None:
        # allow dict-like fallback
        tables = processed.get("tables") if isinstance(processed, dict) else None
    if tables is None:
        raise AttributeError("processed must have a .tables dict (or dict['tables']).")

    # artefacts
    artefacts = getattr(processed, "artefacts", None)
    if artefacts is None:
        artefacts = processed.get("artefacts") if isinstance(processed, dict) else {}

    # meta
    meta = getattr(processed, "meta", None)
    if meta is None:
        meta = processed.get("meta") if isinstance(processed, dict) else {}

    out: Dict[str, Any] = {"splits": {}, "overall": {}, "artefacts": {}}

    # Build overall concat for some global counts (if present)
    dfs: Dict[str, pd.DataFrame] = {}
    for sp in splits:
        if sp in tables and tables[sp] is not None:
            dfs[sp] = _as_df(tables[sp]).copy()

    if not dfs:
        raise ValueError(f"No splits found among {splits}.")

    # Detect columns from the first available split
    first_df = next(iter(dfs.values()))
    subj_col, hadm_col = _detect_id_cols(first_df)
    time_col = _detect_time_col(first_df)

    diag_len_col = _detect_len_col(first_df, "DIAG")
    proc_len_col = _detect_len_col(first_df, "PROC")
    med_len_col = _detect_len_col(first_df, "MED")

    diag_list_col = _detect_codes_or_ids_col(first_df, "DIAG")
    proc_list_col = _detect_codes_or_ids_col(first_df, "PROC")
    med_list_col = _detect_codes_or_ids_col(first_df, "MED")

    # Header
    _print_section("Dataset summary")
    print(f"Splits: {', '.join(dfs.keys())}")
    if isinstance(meta, dict) and "columns" in meta:
        print(f"Meta columns: {len(meta['columns'])}")

    # Per-split stats
    for sp, df in dfs.items():
        n_rows = len(df)
        n_patients = df[subj_col].nunique() if subj_col and subj_col in df.columns else np.nan
        n_adm = df[hadm_col].nunique() if hadm_col and hadm_col in df.columns else np.nan

        # time range
        t_min = t_max = None
        if time_col and time_col in df.columns:
            ts = pd.to_datetime(df[time_col], errors="coerce")
            if ts.notna().any():
                t_min = ts.min()
                t_max = ts.max()

        # lengths: prefer *_LEN columns; else compute from *_IDS/_CODES
        def get_len_series(len_col: Optional[str], list_col: Optional[str]) -> pd.Series:
            if len_col and len_col in df.columns:
                return pd.to_numeric(df[len_col], errors="coerce")
            if list_col and list_col in df.columns:
                return df[list_col].map(_safe_len)
            return pd.Series([np.nan] * len(df))

        diag_l = get_len_series(diag_len_col, diag_list_col)
        proc_l = get_len_series(proc_len_col, proc_list_col)
        med_l = get_len_series(med_len_col, med_list_col)

        split_stats = {
            "rows": int(n_rows),
            "patients": None if np.isnan(n_patients) else int(n_patients),
            "admissions": None if np.isnan(n_adm) else int(n_adm),
            "time_min": None if t_min is None else str(t_min),
            "time_max": None if t_max is None else str(t_max),
            "diag_len": {"mean": float(np.nanmean(diag_l)), **_q(diag_l)},
            "proc_len": {"mean": float(np.nanmean(proc_l)), **_q(proc_l)},
            "med_len": {"mean": float(np.nanmean(med_l)), **_q(med_l)},
        }
        out["splits"][sp] = split_stats

        _print_section(f"{sp.upper()} split")
        print(f"Rows:       {_fmt_int(n_rows)}")
        if split_stats["patients"] is not None:
            print(f"Patients:   {_fmt_int(split_stats['patients'])}")
        if split_stats["admissions"] is not None:
            print(f"Admissions: {_fmt_int(split_stats['admissions'])}")
        if t_min is not None and t_max is not None:
            print(f"Time span:  {t_min.date()} → {t_max.date()}")

        def print_len_block(name: str, stats: Dict[str, float]) -> None:
            mean = _fmt_float(stats["mean"], 2)
            p50 = _fmt_float(stats.get("p50", np.nan), 0)
            p90 = _fmt_float(stats.get("p90", np.nan), 0)
            p95 = _fmt_float(stats.get("p95", np.nan), 0)
            p99 = _fmt_float(stats.get("p99", np.nan), 0)
            mx = _fmt_float(stats.get("p100", np.nan), 0)
            print(f"{name:<10} mean={mean:<8} p50={p50:<4} p90={p90:<4} p95={p95:<4} p99={p99:<4} max={mx}")

        print_len_block("DIAG len", split_stats["diag_len"])
        print_len_block("PROC len", split_stats["proc_len"])
        print_len_block("MED len", split_stats["med_len"])

        # Small examples (first few rows) for sanity: show counts
        example_rows = min(max_unique_examples, n_rows)
        if example_rows > 0:
            ex = df.head(example_rows)
            print("\nExamples (first rows):")
            cols_to_show = []
            for c in (subj_col, hadm_col, time_col):
                if c and c in df.columns:
                    cols_to_show.append(c)
            for c in (diag_list_col, proc_list_col, med_list_col):
                if c and c in df.columns:
                    cols_to_show.append(c)

            if cols_to_show:
                for i in range(example_rows):
                    row = ex.iloc[i]
                    parts = []
                    if subj_col and subj_col in ex.columns:
                        parts.append(f"SUBJECT_ID={row[subj_col]}")
                    if hadm_col and hadm_col in ex.columns:
                        parts.append(f"HADM_ID={row[hadm_col]}")
                    if time_col and time_col in ex.columns:
                        ts = pd.to_datetime(row[time_col], errors="coerce")
                        if pd.notna(ts):
                            parts.append(f"ADMITTIME={ts.date()}")
                    if diag_list_col and diag_list_col in ex.columns:
                        parts.append(f"DIAG={int(_safe_len(row[diag_list_col]))}")
                    if proc_list_col and proc_list_col in ex.columns:
                        parts.append(f"PROC={int(_safe_len(row[proc_list_col]))}")
                    if med_list_col and med_list_col in ex.columns:
                        parts.append(f"MED={int(_safe_len(row[med_list_col]))}")
                    print("  - " + ", ".join(parts))

    # Overall stats across splits
    all_df = pd.concat(list(dfs.values()), ignore_index=True, sort=False)
    overall = {
        "rows": int(len(all_df)),
        "patients": int(all_df[subj_col].nunique()) if subj_col and subj_col in all_df.columns else None,
        "admissions": int(all_df[hadm_col].nunique()) if hadm_col and hadm_col in all_df.columns else None,
    }
    out["overall"] = overall

    _print_section("Overall")
    print(f"Rows:       {_fmt_int(overall['rows'])}")
    if overall["patients"] is not None:
        print(f"Patients:   {_fmt_int(overall['patients'])}")
    if overall["admissions"] is not None:
        print(f"Admissions: {_fmt_int(overall['admissions'])}")

    # Artefacts stats (vocab + adjacency)
    vocab = artefacts.get("vocab") if isinstance(artefacts, dict) else None
    ehr_adj = artefacts.get("ehr_adj") if isinstance(artefacts, dict) else None
    ddi_adj = artefacts.get("ddi_adj") if isinstance(artefacts, dict) else None

    _print_section("Artefacts")
    if vocab is not None:
        # try to infer sizes
        vocab_info = {}
        try:
            if isinstance(vocab, dict):
                for k, v in vocab.items():
                    # common pattern: vocab["diag"], vocab["proc"], vocab["med"]
                    try:
                        vocab_info[k] = len(v[0])
                    except Exception:
                        pass
            else:
                # unknown structure
                vocab_info["vocab"] = len(vocab)  # type: ignore[arg-type]
        except Exception:
            vocab_info = {}
        out["artefacts"]["vocab_sizes"] = vocab_info
        if vocab_info:
            for k, n in vocab_info.items():
                print(f"Vocab {k}: {_fmt_int(n)}")
        else:
            print("Vocab: present (could not infer sizes)")
    else:
        print("Vocab: —")

    def adj_info(name: str, adj: Any) -> Dict[str, Any]:
        if adj is None:
            return {"present": False}
        info: Dict[str, Any] = {"present": True}
        # numpy / torch / scipy sparse support (basic)
        try:
            import torch  # optional

            if isinstance(adj, torch.Tensor):
                info["shape"] = tuple(adj.shape)
                info["nnz"] = int((adj != 0).sum().item())
                return info
        except Exception:
            pass

        if isinstance(adj, np.ndarray):
            info["shape"] = adj.shape
            info["nnz"] = int(np.count_nonzero(adj))
            return info

        # scipy sparse
        try:
            import scipy.sparse as sp  # optional

            if sp.issparse(adj):
                info["shape"] = adj.shape
                info["nnz"] = int(adj.nnz)
                return info
        except Exception:
            pass

        # fallback
        try:
            info["shape"] = tuple(adj.shape)  # type: ignore[attr-defined]
        except Exception:
            pass
        return info

    ehr_info = adj_info("ehr_adj", ehr_adj)
    ddi_info = adj_info("ddi_adj", ddi_adj)
    out["artefacts"]["ehr_adj"] = ehr_info
    out["artefacts"]["ddi_adj"] = ddi_info

    def print_adj(name: str, info: Dict[str, Any]) -> None:
        if not info.get("present"):
            print(f"{name}: —")
            return
        shp = info.get("shape")
        nnz = info.get("nnz")
        if shp is not None and nnz is not None:
            print(f"{name}: shape={shp}, nnz={_fmt_int(nnz)}")
        elif shp is not None:
            print(f"{name}: shape={shp}")
        else:
            print(f"{name}: present")

    print_adj("ehr_adj", ehr_info)
    print_adj("ddi_adj", ddi_info)

    return out