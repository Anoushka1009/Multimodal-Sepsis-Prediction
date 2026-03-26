from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .io import iter_table_chunks, load_table


ID_COLUMNS = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]


def _parse_mimic_datetime(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    remaining = text.notna() & parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(text.loc[remaining], format="%Y-%m-%d", errors="coerce")
    return parsed


def _load_dictionary(extracted_dir: str | Path, filename: str) -> pd.DataFrame:
    try:
        df = load_table(
            extracted_dir=extracted_dir,
            table_name=filename,
            usecols=lambda c: str(c).upper() in {"ITEMID", "LABEL"},
            low_memory=False,
        )
    except FileNotFoundError:
        return pd.DataFrame(columns=["ITEMID", "LABEL"])
    df.columns = [column.upper() for column in df.columns]
    return df.rename(columns={"ITEMID": "ITEMID", "LABEL": "LABEL"})


def resolve_itemids_by_keywords(
    extracted_dir: str | Path,
    filename: str,
    keyword_map: Dict[str, List[str]],
) -> Dict[str, List[int]]:
    dictionary = _load_dictionary(extracted_dir, filename)
    if dictionary.empty:
        return {concept: [] for concept in keyword_map}

    labels = dictionary["LABEL"].astype(str).str.lower()
    resolved: Dict[str, List[int]] = {}
    for concept, keywords in keyword_map.items():
        mask = pd.Series(False, index=dictionary.index)
        for keyword in keywords:
            mask = mask | labels.str.contains(keyword.lower(), na=False, regex=False)
        resolved[concept] = sorted(dictionary.loc[mask, "ITEMID"].dropna().astype(int).unique().tolist())
    return resolved


def attach_icustay_ids(events: pd.DataFrame, cohort: pd.DataFrame, time_column: str = "charttime") -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=ID_COLUMNS + [column for column in events.columns if column not in ID_COLUMNS])

    cohort_slice = cohort[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]].drop_duplicates().copy()
    merged = events.merge(cohort_slice, on=["SUBJECT_ID", "HADM_ID"], how="left", suffixes=("", "_cohort"))
    matched = merged.loc[
        (merged[time_column] >= merged["INTIME"]) & (merged[time_column] <= merged["OUTTIME"])
    ].copy()
    return matched


def detect_antibiotic_orders(
    extracted_dir: str | Path,
    antibiotic_keywords: Iterable[str],
    low_memory: bool = True,
) -> pd.DataFrame:
    prescriptions = load_table(
        extracted_dir=extracted_dir,
        table_name="PRESCRIPTIONS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE", "DRUG", "DRUG_NAME_GENERIC"],
        low_memory=low_memory,
    )
    for column in ["STARTDATE", "ENDDATE"]:
        if column in prescriptions.columns:
            prescriptions[column] = pd.to_datetime(prescriptions[column], errors="coerce")

    drug_text = (
        prescriptions.get("DRUG_NAME_GENERIC", pd.Series(index=prescriptions.index, dtype="object")).fillna("")
        + " "
        + prescriptions.get("DRUG", pd.Series(index=prescriptions.index, dtype="object")).fillna("")
    ).astype(str).str.lower()

    mask = pd.Series(False, index=prescriptions.index)
    for keyword in antibiotic_keywords:
        mask = mask | drug_text.str.contains(keyword.lower(), na=False, regex=False)

    antibiotics = prescriptions.loc[mask, ["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE", "DRUG", "DRUG_NAME_GENERIC"]].copy()
    antibiotics = antibiotics.rename(columns={"STARTDATE": "antibiotic_time"})
    antibiotics = antibiotics.dropna(subset=["antibiotic_time", "HADM_ID"])
    return antibiotics.sort_values(["HADM_ID", "antibiotic_time"]).reset_index(drop=True)


def detect_culture_orders(extracted_dir: str | Path, low_memory: bool = True) -> pd.DataFrame:
    cultures = load_table(
        extracted_dir=extracted_dir,
        table_name="MICROBIOLOGYEVENTS.csv",
        usecols=lambda c: str(c).upper() in {
            "SUBJECT_ID",
            "HADM_ID",
            "CHARTTIME",
            "CHARTDATE",
            "SPEC_TYPE_DESC",
            "ORG_NAME",
            "TEST_NAME",
            "AB_NAME",
            "INTERPRETATION",
        },
        low_memory=low_memory,
    )

    time_column = "CHARTTIME" if "CHARTTIME" in cultures.columns else "CHARTDATE"
    if time_column not in cultures.columns:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "culture_time"])

    cultures[time_column] = _parse_mimic_datetime(cultures[time_column])
    cultures = cultures.dropna(subset=["HADM_ID", time_column]).copy()
    cultures = cultures.rename(columns={time_column: "culture_time"})
    return cultures.sort_values(["HADM_ID", "culture_time"]).reset_index(drop=True)


def derive_suspected_infection(
    antibiotics: pd.DataFrame,
    cultures: pd.DataFrame,
    culture_after_antibiotic_hours: int = 24,
    antibiotic_after_culture_hours: int = 72,
) -> pd.DataFrame:
    if antibiotics.empty or cultures.empty:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "suspected_infection_time", "culture_time", "antibiotic_time"])

    rows = []
    culture_after = pd.Timedelta(hours=culture_after_antibiotic_hours)
    antibiotic_after = pd.Timedelta(hours=antibiotic_after_culture_hours)
    culture_hadm_ids = set(cultures["HADM_ID"].unique().tolist())

    for hadm_id, abx_group in antibiotics.groupby("HADM_ID"):
        if hadm_id not in culture_hadm_ids:
            continue
        culture_group = cultures.loc[cultures["HADM_ID"] == hadm_id]
        abx_times = abx_group.sort_values("antibiotic_time")
        culture_times = culture_group.sort_values("culture_time")

        best_pair = None
        best_time = None
        for _, abx_row in abx_times.iterrows():
            abx_time = abx_row["antibiotic_time"]
            valid = culture_times.loc[
                (culture_times["culture_time"] >= abx_time - antibiotic_after)
                & (culture_times["culture_time"] <= abx_time + culture_after)
            ]
            if valid.empty:
                continue
            candidate = valid.iloc[0]
            suspicion_time = min(abx_time, candidate["culture_time"])
            if best_time is None or suspicion_time < best_time:
                best_time = suspicion_time
                best_pair = (abx_row, candidate, suspicion_time)
        if best_pair is not None:
            abx_row, culture_row, suspicion_time = best_pair
            rows.append(
                {
                    "SUBJECT_ID": int(abx_row["SUBJECT_ID"]),
                    "HADM_ID": int(hadm_id),
                    "suspected_infection_time": suspicion_time,
                    "culture_time": culture_row["culture_time"],
                    "antibiotic_time": abx_row["antibiotic_time"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "suspected_infection_time", "culture_time", "antibiotic_time"])
    return pd.DataFrame(rows).sort_values(["HADM_ID", "suspected_infection_time"]).reset_index(drop=True)


def _extract_item_events(
    extracted_dir: str | Path,
    table_name: str,
    itemids_by_concept: Dict[str, List[int]],
    time_column: str,
    value_column: str,
    chunksize: int,
    low_memory: bool,
) -> pd.DataFrame:
    concept_rows = []
    all_itemids = sorted({itemid for itemids in itemids_by_concept.values() for itemid in itemids})
    if not all_itemids:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "concept", "value"])

    reverse_map = {itemid: concept for concept, itemids in itemids_by_concept.items() for itemid in itemids}
    base_usecols = ["SUBJECT_ID", "HADM_ID", "ITEMID", time_column, value_column]
    if table_name == "CHARTEVENTS.csv":
        base_usecols.insert(2, "ICUSTAY_ID")

    for chunk in iter_table_chunks(
        extracted_dir=extracted_dir,
        table_name=table_name,
        usecols=base_usecols,
        chunksize=chunksize,
        low_memory=low_memory,
    ):
        filtered = chunk.loc[chunk["ITEMID"].isin(all_itemids)].copy()
        if filtered.empty:
            continue
        if "ICUSTAY_ID" not in filtered.columns:
            filtered["ICUSTAY_ID"] = np.nan
        filtered["charttime"] = _parse_mimic_datetime(filtered[time_column])
        filtered["value"] = pd.to_numeric(filtered[value_column], errors="coerce")
        filtered = filtered.dropna(subset=["charttime", "value"])
        filtered["concept"] = filtered["ITEMID"].map(reverse_map)
        concept_rows.append(filtered[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "concept", "value"]])

    if not concept_rows:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "concept", "value"])
    return pd.concat(concept_rows, ignore_index=True)


def extract_sofa_measurements(
    extracted_dir: str | Path,
    cohort: pd.DataFrame,
    lab_item_keywords: Dict[str, List[str]],
    chart_item_keywords: Dict[str, List[str]],
    vasopressor_keywords: Iterable[str],
    chunksize: int = 100000,
    low_memory: bool = True,
) -> Dict[str, pd.DataFrame]:
    lab_itemids = resolve_itemids_by_keywords(extracted_dir, "D_LABITEMS.csv", lab_item_keywords)
    chart_itemids = resolve_itemids_by_keywords(extracted_dir, "D_ITEMS.csv", chart_item_keywords)

    lab_events = _extract_item_events(
        extracted_dir=extracted_dir,
        table_name="LABEVENTS.csv",
        itemids_by_concept=lab_itemids,
        time_column="CHARTTIME",
        value_column="VALUENUM",
        chunksize=chunksize,
        low_memory=low_memory,
    )
    if not lab_events.empty:
        lab_events = attach_icustay_ids(lab_events.drop(columns=["ICUSTAY_ID"]), cohort, time_column="charttime")

    chart_events = _extract_item_events(
        extracted_dir=extracted_dir,
        table_name="CHARTEVENTS.csv",
        itemids_by_concept=chart_itemids,
        time_column="CHARTTIME",
        value_column="VALUENUM",
        chunksize=chunksize,
        low_memory=low_memory,
    )
    if not chart_events.empty:
        missing_icustay = chart_events["ICUSTAY_ID"].isna()
        if missing_icustay.any():
            repaired = attach_icustay_ids(chart_events.loc[missing_icustay].drop(columns=["ICUSTAY_ID"]), cohort, time_column="charttime")
            chart_events = pd.concat([chart_events.loc[~missing_icustay], repaired], ignore_index=True)

    vasopressors = detect_antibiotic_orders(extracted_dir, vasopressor_keywords, low_memory=low_memory)
    if not vasopressors.empty:
        vasopressors = vasopressors.rename(columns={"antibiotic_time": "charttime"})
        vasopressors["concept"] = "vasopressor"
        vasopressors["value"] = 1.0
        vasopressors = attach_icustay_ids(vasopressors[["SUBJECT_ID", "HADM_ID", "charttime", "concept", "value"]], cohort, time_column="charttime")
    else:
        vasopressors = pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "concept", "value", "INTIME", "OUTTIME"])

    return {
        "lab_events": lab_events,
        "chart_events": chart_events,
        "vasopressor_events": vasopressors,
        "lab_itemids": pd.DataFrame([
            {"concept": concept, "itemid": itemid}
            for concept, itemids in lab_itemids.items()
            for itemid in itemids
        ]),
        "chart_itemids": pd.DataFrame([
            {"concept": concept, "itemid": itemid}
            for concept, itemids in chart_itemids.items()
            for itemid in itemids
        ]),
    }


def _score_platelet(value: float) -> int:
    if value < 20:
        return 4
    if value < 50:
        return 3
    if value < 100:
        return 2
    if value < 150:
        return 1
    return 0


def _score_bilirubin(value: float) -> int:
    if value >= 12:
        return 4
    if value >= 6:
        return 3
    if value >= 2:
        return 2
    if value >= 1.2:
        return 1
    return 0


def _score_creatinine(value: float) -> int:
    if value >= 5:
        return 4
    if value >= 3.5:
        return 3
    if value >= 2:
        return 2
    if value >= 1.2:
        return 1
    return 0


def _score_map(value: float) -> int:
    return 1 if value < 70 else 0


def _score_gcs(value: float) -> int:
    if value < 6:
        return 4
    if value < 10:
        return 3
    if value < 13:
        return 2
    if value < 15:
        return 1
    return 0


def _score_pafi(value: float) -> int:
    if value < 100:
        return 4
    if value < 200:
        return 3
    if value < 300:
        return 2
    if value < 400:
        return 1
    return 0


def compute_sofa_hourly(measurements: Dict[str, pd.DataFrame], cohort: pd.DataFrame) -> pd.DataFrame:
    lab_events = measurements.get("lab_events", pd.DataFrame()).copy()
    chart_events = measurements.get("chart_events", pd.DataFrame()).copy()
    vasopressor_events = measurements.get("vasopressor_events", pd.DataFrame()).copy()

    all_events = pd.concat([lab_events, chart_events, vasopressor_events], ignore_index=True)
    if all_events.empty:
        return pd.DataFrame(columns=ID_COLUMNS + ["hour", "respiratory", "coagulation", "liver", "cardiovascular", "cns", "renal", "total_sofa"])

    if "INTIME" in all_events.columns:
        all_events = all_events.drop(columns=[column for column in ["INTIME", "OUTTIME"] if column in all_events.columns])

    all_events["hour"] = pd.to_datetime(all_events["charttime"], errors="coerce").dt.floor("h")
    all_events = all_events.dropna(subset=["ICUSTAY_ID", "hour"]).copy()

    rows = []
    for key, group in all_events.groupby(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "hour"], dropna=False):
        subject_id, hadm_id, icustay_id, hour = key
        concept_map = {concept: concept_df["value"].astype(float).tolist() for concept, concept_df in group.groupby("concept")}

        respiratory = 0
        if "pao2" in concept_map and "fio2" in concept_map:
            fio2 = float(np.nanmean(concept_map["fio2"]))
            pafi = float(np.nanmean(concept_map["pao2"]) / fio2) if fio2 not in [0.0] and not np.isnan(fio2) else np.nan
            respiratory = _score_pafi(pafi) if not np.isnan(pafi) else 0

        coagulation = max([_score_platelet(v) for v in concept_map.get("platelet", [np.inf])] or [0])
        liver = max([_score_bilirubin(v) for v in concept_map.get("bilirubin", [-np.inf])] or [0])
        renal = max([_score_creatinine(v) for v in concept_map.get("creatinine", [-np.inf])] or [0])
        cardiovascular = max(
            max([_score_map(v) for v in concept_map.get("map", [np.inf])] or [0]),
            3 if "vasopressor" in concept_map else 0,
        )
        cns = max([_score_gcs(v) for v in concept_map.get("gcs_total", [15])] or [0])

        rows.append(
            {
                "SUBJECT_ID": subject_id,
                "HADM_ID": hadm_id,
                "ICUSTAY_ID": icustay_id,
                "hour": hour,
                "respiratory": respiratory,
                "coagulation": coagulation,
                "liver": liver,
                "cardiovascular": cardiovascular,
                "cns": cns,
                "renal": renal,
            }
        )

    sofa = pd.DataFrame(rows)
    component_columns = ["respiratory", "coagulation", "liver", "cardiovascular", "cns", "renal"]
    sofa[component_columns] = sofa[component_columns].fillna(0)
    sofa["total_sofa"] = sofa[component_columns].sum(axis=1)
    sofa = sofa.sort_values(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "hour"]).reset_index(drop=True)
    return sofa


def derive_sepsis_labels(
    cohort: pd.DataFrame,
    suspected_infection: pd.DataFrame,
    sofa_hourly: pd.DataFrame,
    baseline_window_hours: int = 24,
    sofa_delta_threshold: int = 2,
    sofa_before_suspicion_hours: int = 48,
    sofa_after_suspicion_hours: int = 24,
) -> pd.DataFrame:
    label_rows = []
    suspected_map = suspected_infection.groupby("HADM_ID")["suspected_infection_time"].min().to_dict() if not suspected_infection.empty else {}

    for _, stay in cohort.iterrows():
        stay_sofa = sofa_hourly.loc[
            (sofa_hourly["SUBJECT_ID"] == stay["SUBJECT_ID"])
            & (sofa_hourly["HADM_ID"] == stay["HADM_ID"])
            & (sofa_hourly["ICUSTAY_ID"] == stay["ICUSTAY_ID"])
        ].copy()
        suspicion_time = suspected_map.get(stay["HADM_ID"], pd.NaT)
        baseline = 0.0
        onset_time = pd.NaT
        max_delta = 0.0

        if not stay_sofa.empty:
            baseline_end = stay["INTIME"] + pd.Timedelta(hours=baseline_window_hours)
            baseline_slice = stay_sofa.loc[
                (stay_sofa["hour"] >= stay["INTIME"]) & (stay_sofa["hour"] <= baseline_end)
            ]
            baseline = float(baseline_slice["total_sofa"].min()) if not baseline_slice.empty else float(stay_sofa["total_sofa"].iloc[0])
            stay_sofa["delta_sofa"] = stay_sofa["total_sofa"] - baseline
            max_delta = float(stay_sofa["delta_sofa"].max()) if not stay_sofa.empty else 0.0

            if pd.notna(suspicion_time):
                candidate = stay_sofa.loc[
                    (stay_sofa["hour"] >= suspicion_time - pd.Timedelta(hours=sofa_before_suspicion_hours))
                    & (stay_sofa["hour"] <= suspicion_time + pd.Timedelta(hours=sofa_after_suspicion_hours))
                    & (stay_sofa["delta_sofa"] >= sofa_delta_threshold)
                ].sort_values("hour")
                if not candidate.empty:
                    onset_time = candidate["hour"].iloc[0]

        label_rows.append(
            {
                "SUBJECT_ID": stay["SUBJECT_ID"],
                "HADM_ID": stay["HADM_ID"],
                "ICUSTAY_ID": stay["ICUSTAY_ID"],
                "suspected_infection_time": suspicion_time,
                "baseline_sofa": baseline,
                "max_sofa_delta": max_delta,
                "sepsis_onset_time": onset_time,
                "sepsis3_label": int(pd.notna(onset_time)),
            }
        )

    return pd.DataFrame(label_rows).sort_values(["SUBJECT_ID", "ICUSTAY_ID"]).reset_index(drop=True)
