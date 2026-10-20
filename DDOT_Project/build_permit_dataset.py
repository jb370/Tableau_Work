"""
DDOT Public Space Construction Permits -> Tableau-ready dataset  (v2)
=====================================================================

  python build_permit_dataset.py --explore   # inspect FY26 data, decide the open questions
  python build_permit_dataset.py             # build output/permits_fy26_clean.csv

Flags:
  --refresh   re-pull from the service instead of using output/permits_raw.csv
  --full      pull the whole layer (where=1=1) if the server-side date filter errors

Requires: pip install pandas numpy requests
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
LAYER_URL = ("https://maps2.dcgis.dc.gov/dcgis/rest/services/"
             "DCGIS_APPS/DDOT_TOPS/MapServer/23")

# Pull only the columns we need (skips big text blobs and name fields).
OUT_FIELDS = [
    "OBJECTID", "TrackingNumber", "PermitNumber", "Status", "StatusDescription",
    "IntakeDate", "ApplicationDate", "ReadyForReviewDate", "AssignedDate",
    "LastApprovalDate", "IssueDate", "PermitFeesPaymentDate",
    "LastResubmissionDate", "LastResubmissionRequestDate", "PCReviewStatus",
    "SourcePermit", "Renewal", "RenewalDescription",
    "PermitType", "PermitTypeDescription",
    "IsExcavation", "IsPaving", "IsFixture", "IsOverHeadWork", "IsOccupancy",
    "IsProjections", "IsLandscaping", "IsPSRental", "IsAnnualPermit",
    "TypeDetailNames", "FiscalYear", "Ward", "WorkLocationWard", "Quadrant",
    "NeighborhoodClusters", "ApplicantCompany",
    "DaysToAssign", "DaysToApprove", "DaysToIssue",
]

# Server-side prefilter (small buffer before Oct 1 because Esri stores UTC).
# Exact filtering happens locally afterward.
PULL_WHERE = "ApplicationDate >= TIMESTAMP '2025-09-25 00:00:00'"

FY_START = pd.Timestamp("2025-10-01")
FY_END = pd.Timestamp("2026-09-30")

# Which date ends the clock for the headline metric?
#   "LastApprovalDate" -> time the District took to approve (my recommendation)
#   "IssueDate"        -> includes applicant time to pay fees after approval
# Both are always calculated; this picks which one becomes `business_days_to_approve`.
PRIMARY_END = "LastApprovalDate"

# "New" permit rule (confirmed in --explore): RenewalDescription == "New"
# (Renewal == 1). Renewals are Renewal == 2 and always carry a SourcePermit.
# PermitType is NOT used: renewals also show PermitType = New.

# Type flags are NOT mutually exclusive. Order = priority for the single
# "primary" category when a permit has several flags.
TYPE_FLAGS = {
    "IsExcavation": "Excavation",
    "IsPaving": "Paving",
    "IsFixture": "Fixture",
    "IsOverHeadWork": "Overhead Work",
    "IsOccupancy": "Occupancy",
    "IsProjections": "Projections",
    "IsLandscaping": "Landscaping",
    "IsPSRental": "Public Space Rental",
}

# Excavation is ~96% of permits, so it is split by work type using keywords in
# TypeDetailNames. First match wins (case-insensitive). Refine after reading the
# "Unclassified excavation detail" printout at the end of the build.
EXCAVATION_RULES = [
    (["gas"], "Gas"),
    (["electric"], "Electric"),
    (["cathodic"], "Cathodic Protection"),
    (["hydrant", "valve", "water ml", "water main", "water lateral conn"], "Water Main / Hydrant / Valve"),
    (["water service lateral", "boring test pit"], "Water Service Lateral / Test Pit"),
    (["multiple house connections"], "Utility House Connections"),
    (["telecom", "fiber", "communication", "conduit"], "Telecom / Conduit"),
]

# Applicant grouping (one applicant, DCWater/CIP, is ~80% of volume).
APPLICANT_RULES = [
    (["dcwater", "dc water"], "DC Water"),
    (["wash gas", "washington gas"], "Washington Gas"),
    (["pepco", "potomac electric"], "Pepco"),
    (["verizon", "comcast", "teleport", "at&t", "lumen", "crown castle", "zayo"], "Telecom"),
    (["department of", "ddot", "dgs", "district of columbia", "dc gov"], "Government"),
]

# DC legal holidays, FY2026. Jul 4, 2026 is a Saturday -> observed Fri Jul 3.
DC_HOLIDAYS = [
    "2025-10-13", "2025-11-11", "2025-11-27", "2025-12-25", "2026-01-01",
    "2026-01-19", "2026-02-16", "2026-04-16", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07",
]
HOL = np.array(DC_HOLIDAYS, dtype="datetime64[D]")

LOCAL_TZ = "America/New_York"
OUT_DIR = Path("output")


# --------------------------------------------------------------------------
# Pull
# --------------------------------------------------------------------------
def get_meta():
    r = requests.get(LAYER_URL, params={"f": "json"}, timeout=60)
    r.raise_for_status()
    meta = r.json()
    if "error" in meta:
        sys.exit(f"Service error: {meta['error']}")
    return meta


def fetch_all(meta, fields, where):
    page = min(int(meta.get("maxRecordCount", 1000)), 2000)
    oid = next((f["name"] for f in meta["fields"] if f["type"] == "esriFieldTypeOID"), "OBJECTID")
    rows, offset = [], 0
    while True:
        params = {
            "where": where, "outFields": ",".join(fields), "returnGeometry": "false",
            "orderByFields": oid, "f": "json",
            "resultOffset": offset, "resultRecordCount": page,
        }
        for attempt in range(3):
            try:
                r = requests.post(f"{LAYER_URL}/query", data=params, timeout=180)
                r.raise_for_status()
                payload = r.json()
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                break
            except Exception as e:
                if attempt == 2:
                    sys.exit(f"Query failed: {e}\nTry again with --full to skip the server-side filter.")
                print(f"  retrying after: {e}")
                time.sleep(2 * (attempt + 1))
        feats = payload.get("features", [])
        rows.extend(f["attributes"] for f in feats)
        print(f"  fetched {len(rows):,} records", end="\r")
        if not feats or not payload.get("exceededTransferLimit", False):
            break
        offset += len(feats)
    print()
    return pd.DataFrame(rows)


def load_raw(refresh=False, full=False):
    OUT_DIR.mkdir(exist_ok=True)
    meta = get_meta()
    avail = {f["name"]: f for f in meta["fields"]}
    fields = [f for f in OUT_FIELDS if f in avail]
    missing = [f for f in OUT_FIELDS if f not in avail]
    if missing:
        print("Fields not found on the layer (skipped):", missing)
    date_fields = [f for f in fields if avail[f]["type"] == "esriFieldTypeDate"]
    raw_path = OUT_DIR / "permits_raw.csv"

    if raw_path.exists() and not refresh:
        print(f"Using cached {raw_path} (use --refresh to re-pull)")
        header = pd.read_csv(raw_path, nrows=0).columns
        return pd.read_csv(raw_path, parse_dates=[c for c in date_fields if c in header],
                           low_memory=False), date_fields

    where = "1=1" if full else PULL_WHERE
    print(f"Pulling from service with where: {where}")
    df = fetch_all(meta, fields, where)
    for c in date_fields:
        s = pd.to_datetime(df[c], unit="ms", utc=True, errors="coerce")
        df[c] = s.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    df.to_csv(raw_path, index=False)
    print(f"Saved {len(df):,} rows to {raw_path}")
    return df, date_fields


# --------------------------------------------------------------------------
# Explore
# --------------------------------------------------------------------------
def explore(df, date_fields):
    pd.set_option("display.width", 220, "display.max_columns", None)
    print(f"\nRows pulled: {len(df):,}")

    print("\n=== DATE COLUMNS: range and % null ===")
    for c in date_fields:
        s = df[c]
        print(f"  {c:30s} {str(s.min())[:10]} -> {str(s.max())[:10]}   null: {s.isna().mean():.0%}")

    fy = df[df["ApplicationDate"] >= FY_START].copy()
    print(f"\n=== APPLICATIONS ON/AFTER {FY_START.date()}: {len(fy):,} rows; "
          f"with IssueDate: {fy['IssueDate'].notna().sum():,} ===")
    iss = fy[fy["IssueDate"].between(FY_START, FY_END)]

    print("\n--- StatusDescription (issued-in-FY26 rows) ---")
    print(iss["StatusDescription"].value_counts(dropna=False).to_string())

    print("\n--- 'New' definition candidates (issued-in-FY26 rows) ---")
    if {"Renewal", "RenewalDescription"} <= set(iss.columns):
        print(pd.crosstab(iss["Renewal"].fillna(-1), iss["RenewalDescription"].fillna("(null)")))
    if "PermitTypeDescription" in iss:
        print("\nPermitTypeDescription:")
        print(iss["PermitTypeDescription"].value_counts(dropna=False).to_string())
    if "SourcePermit" in iss:
        sp = iss["SourcePermit"].notna() & (iss["SourcePermit"].astype(str).str.strip() != "")
        print("\nHas SourcePermit, by Renewal value:")
        print(pd.crosstab(iss["Renewal"].fillna(-1), sp, colnames=["has_SourcePermit"]))

    print("\n--- Type flags (issued-in-FY26 rows) ---")
    flags = [c for c in TYPE_FLAGS if c in iss]
    print(iss[flags].fillna(0).astype(int).sum().to_string())
    combo = iss[flags].fillna(0).astype(int).astype(str).agg("".join, axis=1)
    print("\nMost common flag combinations (order:", ",".join(flags), "):")
    print(combo.value_counts().head(15).to_string())
    if "IsAnnualPermit" in iss:
        print("\nIsAnnualPermit:", iss["IsAnnualPermit"].fillna(0).value_counts().to_dict())
    if "TypeDetailNames" in iss:
        print("\nTop TypeDetailNames:")
        print(iss["TypeDetailNames"].value_counts().head(20).to_string())

    print("\n--- Which end date? LastApprovalDate vs IssueDate ---")
    print("LastApprovalDate null:", f"{iss['LastApprovalDate'].isna().mean():.0%}",
          "| equals IssueDate:", f"{(iss['LastApprovalDate'] == iss['IssueDate']).mean():.0%}",
          "| after IssueDate:", f"{(iss['LastApprovalDate'] > iss['IssueDate']).mean():.0%}")
    gap = (iss["IssueDate"] - iss["LastApprovalDate"]).dt.days
    print("Calendar days approval->issue:\n", gap.describe().to_string())

    print("\n--- Other segment candidates ---")
    for c in ["FiscalYear", "Ward", "WorkLocationWard", "Quadrant", "PCReviewStatus"]:
        if c in iss:
            print(f"\n{c}:")
            print(iss[c].value_counts(dropna=False).head(15).to_string())
    if "LastResubmissionDate" in iss:
        print("\n% with a resubmission:", f"{iss['LastResubmissionDate'].notna().mean():.0%}")
    if "ApplicantCompany" in iss:
        print("\nTop ApplicantCompany:")
        print(iss["ApplicantCompany"].value_counts().head(15).to_string())

    print("\n=== SAMPLE (3 rows) ===")
    print(iss.sample(min(3, len(iss)), random_state=1).T.to_string())
    print("\nNext: spot-check 5-10 permits in the TOPS lookup, then run without --explore.")


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def bdays(start, end):
    """Business days from start (inclusive) to end (exclusive), excluding
    weekends and DC holidays. Same-day = 0. NaN if either date is missing
    or the interval is negative."""
    out = pd.Series(np.nan, index=start.index, dtype="float64")
    ok = start.notna() & end.notna()
    if ok.any():
        out.loc[ok] = np.busday_count(
            start[ok].values.astype("datetime64[D]"),
            end[ok].values.astype("datetime64[D]"),
            holidays=HOL,
        )
    return out.where(~(out < 0))


def is_new_permit(d):
    return d["RenewalDescription"].astype(str).str.strip().str.lower().eq("new")


def build(df):
    d = df.copy()
    log = [("Raw records pulled", len(d))]

    if "Status" in d:  # -4 = Trash/Delete
        d = d[d["Status"].fillna(0) != -4]
        log.append(("Excluded Trash/Delete status", len(d)))

    d = d[d["ApplicationDate"] >= FY_START]
    log.append((f"ApplicationDate >= {FY_START.date()}", len(d)))

    d = d[d["IssueDate"].notna() & d["IssueDate"].between(FY_START, FY_END)]
    log.append(("Issued in FY26 (IssueDate in window)", len(d)))

    d = d[is_new_permit(d)]
    log.append(("New permits only (RenewalDescription == New)", len(d)))

    if "PermitNumber" in d:
        has = d["PermitNumber"].notna()
        dup = d[has].duplicated("PermitNumber", keep="first")
        if dup.any():
            print(f"NOTE: dropping {dup.sum():,} duplicate PermitNumber rows.")
            d = d.drop(dup[dup].index)
        log.append(("Deduplicated on PermitNumber", len(d)))

    # ---- business-day measures
    d["bd_application_to_approval"] = bdays(d["ApplicationDate"], d["LastApprovalDate"])
    d["bd_application_to_issue"] = bdays(d["ApplicationDate"], d["IssueDate"])
    primary = "bd_application_to_approval" if PRIMARY_END == "LastApprovalDate" else "bd_application_to_issue"
    d["business_days_to_approve"] = d[primary]
    d["calendar_days_application_to_approval"] = (d["LastApprovalDate"] - d["ApplicationDate"]).dt.days

    # ---- stage durations ("where does the time go")
    d["bd_stage_1_app_to_ready"] = bdays(d["ApplicationDate"], d["ReadyForReviewDate"])
    d["bd_stage_2_ready_to_assigned"] = bdays(d["ReadyForReviewDate"], d["AssignedDate"])
    d["bd_stage_3_assigned_to_approval"] = bdays(d["AssignedDate"], d["LastApprovalDate"])
    d["bd_stage_4_approval_to_issue"] = bdays(d["LastApprovalDate"], d["IssueDate"])

    n_null = d["business_days_to_approve"].isna().sum()
    log.append((f"Rows with usable {PRIMARY_END}-based metric (dropped from averages: {n_null})",
                int(d["business_days_to_approve"].notna().sum())))

    # ---- segmentation fields
    flag_cols = [c for c in TYPE_FLAGS if c in d]
    M = d[flag_cols].fillna(0).astype(int).gt(0).values
    names = np.array([TYPE_FLAGS[c] for c in flag_cols])
    d["n_type_flags"] = M.sum(axis=1)
    d["permit_flag_primary"] = [names[r][0] if r.any() else "Other / Unflagged" for r in M]
    d["permit_type_combo"] = [" + ".join(names[r]) if r.any() else "Other / Unflagged" for r in M]

    detail = (d["TypeDetailNames"].fillna("").str.lower() if "TypeDetailNames" in d
              else pd.Series("", index=d.index))

    def first_match(text, rules, default):
        for kws, label in rules:
            if any(k in text for k in kws):
                return label
        return default

    is_exc = d["IsExcavation"].fillna(0).gt(0) if "IsExcavation" in d else pd.Series(False, index=d.index)
    d["excavation_work_type"] = np.where(
        is_exc, detail.map(lambda t: first_match(t, EXCAVATION_RULES, "Other Excavation")), "Not excavation")
    d["permit_type_category"] = np.where(
        is_exc, "Excavation: " + d["excavation_work_type"], d["permit_flag_primary"])

    comp = (d["ApplicantCompany"].fillna("").str.lower() if "ApplicantCompany" in d
            else pd.Series("", index=d.index))
    d["applicant_group"] = comp.map(lambda t: first_match(t, APPLICANT_RULES, "Other") if t else "Unknown")

    oth = d[d["excavation_work_type"] == "Other Excavation"]
    if len(oth):
        print(f"\nUnclassified excavation detail ({len(oth):,} permits; top 15) - add keywords to EXCAVATION_RULES:")
        print(oth["TypeDetailNames"].value_counts().head(15).to_string())

    w = pd.to_numeric(d["Ward"], errors="coerce") if "Ward" in d else pd.Series(np.nan, index=d.index)
    if "WorkLocationWard" in d:
        w = w.fillna(pd.to_numeric(d["WorkLocationWard"], errors="coerce"))
    d["ward"] = w.map(lambda x: "Unknown" if pd.isna(x) else f"Ward {int(x)}")

    d["review_path"] = np.where(d["LastResubmissionDate"].notna(), "Resubmitted", "First submission") \
        if "LastResubmissionDate" in d else "Unknown"
    d["application_month"] = d["ApplicationDate"].dt.to_period("M").dt.to_timestamp()
    d["application_weekday"] = d["ApplicationDate"].dt.day_name()
    d["wait_bucket"] = pd.cut(
        d["business_days_to_approve"], bins=[-1, 0, 5, 10, 20, 40, np.inf],
        labels=["Same day", "1-5", "6-10", "11-20", "21-40", "40+"],
    ).astype(object).where(d["business_days_to_approve"].notna(), "No approval date")

    # ---- sanity check vs the service's own DaysToApprove
    if "DaysToApprove" in d:
        diff = (d["DaysToApprove"] - d["calendar_days_application_to_approval"]).dropna()
        if len(diff):
            print("\nCheck vs service DaysToApprove (calendar-day diff): "
                  f"exact match {(diff == 0).mean():.0%}, median diff {diff.median():.0f}. "
                  "If match is low, DaysToApprove may be business days or use a different start/end.")

    return d, pd.DataFrame(log, columns=["step", "rows_remaining"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--check-count", action="store_true", help="compare cached rows to the server's record count")
    a = ap.parse_args()

    df, date_fields = load_raw(refresh=a.refresh, full=a.full)
    if a.check_count:
        r = requests.post(f"{LAYER_URL}/query", timeout=60, data={
            "where": "1=1" if a.full else PULL_WHERE, "returnCountOnly": "true", "f": "json"})
        print(f"Server count: {r.json().get('count')} | cached rows: {len(df)}")
        return
    if a.explore:
        explore(df, date_fields)
        return
    d, log = build(df)
    OUT_DIR.mkdir(exist_ok=True)
    d.to_csv(OUT_DIR / "permits_fy26_clean.csv", index=False)
    log.to_csv(OUT_DIR / "excluded_log.csv", index=False)
    print(f"\nSaved {len(d):,} rows to {OUT_DIR / 'permits_fy26_clean.csv'}\n")
    print(log.to_string(index=False))
    m = d["business_days_to_approve"]
    print(f"\nOverall: mean {m.mean():.1f} | median {m.median():.1f} business days (n={m.notna().sum():,})")
    for seg in ["permit_type_category", "applicant_group", "ward", "review_path"]:
        print(f"\n--- by {seg} ---")
        print(d.groupby(seg)["business_days_to_approve"]
                .agg(["count", "mean", "median"]).round(1).sort_values("count", ascending=False).to_string())


if __name__ == "__main__":
    main()
