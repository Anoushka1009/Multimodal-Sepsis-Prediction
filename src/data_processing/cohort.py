from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .io import load_table


ID_COLUMNS = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]
TIME_COLUMNS = ["ADMITTIME", "DISCHTIME", "INTIME", "OUTTIME", "DOB", "DOD"]


def _parse_datetimes(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def _compute_age_years(intime: pd.Series, dob: pd.Series) -> pd.Series:
    intime = pd.to_datetime(intime, errors="coerce")
    dob = pd.to_datetime(dob, errors="coerce")

    age = (intime.dt.year - dob.dt.year).astype("float64")
    before_birthday = (
        (intime.dt.month < dob.dt.month)
        | ((intime.dt.month == dob.dt.month) & (intime.dt.day < dob.dt.day))
    )
    age = age - before_birthday.fillna(False).astype("float64")
    age = age.where(intime.notna() & dob.notna())
    return age


def build_base_icu_cohort(
    extracted_dir: str | Path,
    adult_age_min: int = 18,
    min_icu_los_hours: float = 6.0,
    first_icu_only: bool = False,
    low_memory: bool = True,
) -> pd.DataFrame:
    patients = load_table(
        extracted_dir=extracted_dir,
        table_name="PATIENTS.csv",
        usecols=["SUBJECT_ID", "GENDER", "DOB", "DOD"],
        low_memory=low_memory,
    )
    admissions = load_table(
        extracted_dir=extracted_dir,
        table_name="ADMISSIONS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "ETHNICITY"],
        low_memory=low_memory,
    )
    icustays = load_table(
        extracted_dir=extracted_dir,
        table_name="ICUSTAYS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "FIRST_CAREUNIT", "LAST_CAREUNIT", "INTIME", "OUTTIME"],
        low_memory=low_memory,
    )

    patients = _parse_datetimes(patients, ["DOB", "DOD"])
    admissions = _parse_datetimes(admissions, ["ADMITTIME", "DISCHTIME", "DEATHTIME"])
    icustays = _parse_datetimes(icustays, ["INTIME", "OUTTIME"])

    cohort = icustays.merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="left")
    cohort = cohort.merge(patients, on=["SUBJECT_ID"], how="left")

    cohort["icu_los_hours"] = (cohort["OUTTIME"] - cohort["INTIME"]).dt.total_seconds() / 3600.0
    cohort["age_at_icu_intime_raw"] = _compute_age_years(cohort["INTIME"], cohort["DOB"])
    cohort["age_is_masked_89_plus"] = cohort["age_at_icu_intime_raw"] >= 300
    cohort["age_at_icu_intime"] = cohort["age_at_icu_intime_raw"].where(~cohort["age_is_masked_89_plus"], 90.0)
    cohort["is_adult_icu"] = (cohort["age_at_icu_intime_raw"] >= adult_age_min) | cohort["age_is_masked_89_plus"]

    cohort = cohort.loc[cohort["is_adult_icu"]].copy()
    cohort = cohort.loc[cohort["icu_los_hours"] >= float(min_icu_los_hours)].copy()
    cohort = cohort.sort_values(["SUBJECT_ID", "INTIME", "ICUSTAY_ID"]).reset_index(drop=True)

    if first_icu_only:
        cohort = cohort.groupby("SUBJECT_ID", as_index=False).head(1).reset_index(drop=True)

    return cohort


def summarize_cohort(cohort: pd.DataFrame) -> Dict[str, float]:
    masked_age = cohort["age_is_masked_89_plus"] if "age_is_masked_89_plus" in cohort else pd.Series(False, index=cohort.index)
    return {
        "icu_stay_count": int(cohort["ICUSTAY_ID"].nunique()) if "ICUSTAY_ID" in cohort else 0,
        "patient_count": int(cohort["SUBJECT_ID"].nunique()) if "SUBJECT_ID" in cohort else 0,
        "admission_count": int(cohort["HADM_ID"].nunique()) if "HADM_ID" in cohort else 0,
        "median_icu_los_hours": float(cohort["icu_los_hours"].median()) if "icu_los_hours" in cohort and not cohort.empty else 0.0,
        "median_age_years": float(cohort.loc[~masked_age, "age_at_icu_intime"].median()) if "age_at_icu_intime" in cohort and not cohort.empty else 0.0,
        "masked_age_89_plus_count": int(cohort["age_is_masked_89_plus"].sum()) if "age_is_masked_89_plus" in cohort else 0,
    }


def create_patient_level_splits(
    cohort: pd.DataFrame,
    val_size: float,
    test_size: float,
    random_state: int = 42,
) -> pd.DataFrame:
    subjects = cohort["SUBJECT_ID"].dropna().astype(int).drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_state)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_test = int(round(n_subjects * test_size))
    n_val = int(round(n_subjects * val_size))
    n_train = max(n_subjects - n_val - n_test, 0)

    train_subjects = set(subjects[:n_train].tolist())
    val_subjects = set(subjects[n_train:n_train + n_val].tolist())
    test_subjects = set(subjects[n_train + n_val:].tolist())

    splits = cohort[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]].drop_duplicates().copy()
    splits["split"] = "train"
    splits.loc[splits["SUBJECT_ID"].isin(val_subjects), "split"] = "val"
    splits.loc[splits["SUBJECT_ID"].isin(test_subjects), "split"] = "test"
    return splits.sort_values(["split", "SUBJECT_ID", "ICUSTAY_ID"]).reset_index(drop=True)
