#!/usr/bin/env python3
"""
GBIF image downloader for the Native Wildlife Badge app.

Pulls openly-licensed photos from GBIF (most are sourced from iNaturalist) for:
  - the TARGET species (positives)            -> data/target/
  - a set of confusable / co-habitat species  -> data/not_target/  (negatives)

Every image's license + attribution is logged to data/licenses.csv, so you can
credit CC-BY photographers and prove you had the right to use each photo.

Setup + run:
    pip install requests
    python download_gbif_images.py

Notes:
  * GBIF stores image URLs, not the images themselves, so this does two passes:
    (1) query occurrence records to collect image URLs, (2) download the files.
  * Positives are NOT restricted to Tampa on purpose — an ibis looks the same
    everywhere, so more geography = more training data. Locality only shaped the
    NEGATIVE_SPECIES list below (the look-alikes a Tampa user might snap instead).
"""

import csv
import hashlib
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------------------------------------------------
# CONFIG  — edit this block, then just run the script
# ----------------------------------------------------------------------

TARGET_SPECIES = "Eudocimus albus"        # White Ibis — your app's target

# Confusable / co-habitat birds a Tampa user might photograph by mistake.
# These teach the model "similar, but NOT an ibis."
NEGATIVE_SPECIES = [
    "Ardea alba",          # Great Egret        (big white wader)
    "Egretta thula",       # Snowy Egret        (white wader)
    "Bubulcus ibis",       # Cattle Egret       (white, common on lawns)
    "Mycteria americana",  # Wood Stork         (white + black, bald head, curved bill)
    "Platalea ajaja",      # Roseate Spoonbill  (pink wader, similar feeding behaviour)
    "Ardea herodias",      # Great Blue Heron   (common co-habitant)
]

MAX_TARGET_IMAGES = 1500          # positives. Start modest; raise once the pipeline works.
COUNTRY = "US"                   # None = worldwide. US keeps the relevant populations.
LICENSES = ["CC0_1_0", "CC_BY_4_0"]   # safe to redistribute.
                                 # (Add "CC_BY_NC_4_0" ONLY if your app stays non-commercial.)
ONLY_WILD = True                 # True -> human observations only (skips museum specimens)

OUT_DIR = "data"
DOWNLOAD_WORKERS = 8             # parallel image downloads
GBIF_API = "https://api.gbif.org/v1/"
HEADERS = {"User-Agent": "WildlifeBadgeApp/0.1 (training data collection)"}

# ----------------------------------------------------------------------
# (you usually don't need to edit below here)
# ----------------------------------------------------------------------


def resolve_taxon_key(name):
    """Turn a scientific name into GBIF's numeric taxonKey (robust matching)."""
    r = requests.get(GBIF_API + "species/match", params={"name": name},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    key = data.get("usageKey")
    if not key or data.get("matchType") == "NONE":
        raise ValueError(f"GBIF couldn't match species name: {name!r}")
    print(f"  matched {name!r} -> taxonKey {key} ({data.get('scientificName')})")
    return key


def fetch_image_records(name, max_images):
    """Page through GBIF occurrences and collect image URLs + attribution."""
    params = {
        "taxonKey": resolve_taxon_key(name),
        "mediaType": "StillImage",
        "license": LICENSES,          # repeated -> OR of these licenses
        "limit": 300,                 # 300 is GBIF's max page size
        "offset": 0,
    }
    if COUNTRY:
        params["country"] = COUNTRY
    if ONLY_WILD:
        params["basisOfRecord"] = "HUMAN_OBSERVATION"

    records = []
    while len(records) < max_images:
        r = requests.get(GBIF_API + "occurrence/search", params=params,
                         headers=HEADERS, timeout=60)
        r.raise_for_status()
        payload = r.json()
        for occ in payload.get("results", []):
            occ_id = occ.get("key", "noid")
            for media in occ.get("media", []):
                url = media.get("identifier")
                if not url:
                    continue
                records.append({
                    "species": name,
                    "occurrence_id": occ_id,
                    "url": url,
                    "license": media.get("license") or occ.get("license", ""),
                    "rights_holder": media.get("rightsHolder", ""),
                    "creator": media.get("creator", ""),
                })
                if len(records) >= max_images:
                    break
            if len(records) >= max_images:
                break
        if payload.get("endOfRecords"):
            break
        params["offset"] += params["limit"]

    print(f"  collected {len(records)} image URLs for {name!r}")
    return records


def download_one(rec, dest_dir, label):
    """Download a single image; return a log row dict, or None on skip/failure."""
    safe = rec["species"].lower().replace(" ", "_")
    url_hash = hashlib.md5(rec["url"].encode()).hexdigest()[:8]   # stable across re-runs
    path = os.path.join(dest_dir, f"{safe}_{rec['occurrence_id']}_{url_hash}.jpg")
    if os.path.exists(path):
        return None
    try:
        resp = requests.get(rec["url"], headers=HEADERS, timeout=30, stream=True)
        resp.raise_for_status()
        if "image" not in resp.headers.get("Content-Type", ""):
            return None
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
    except Exception:
        return None   # dead link / timeout — just skip it
    return {
        "filepath": path, "label": label, "species": rec["species"],
        "occurrence_id": rec["occurrence_id"], "url": rec["url"],
        "license": rec["license"], "rights_holder": rec["rights_holder"],
        "creator": rec["creator"],
    }


def download_records(records, label):
    """Download a batch of image records into data/<label>/ ."""
    dest_dir = os.path.join(OUT_DIR, label)
    os.makedirs(dest_dir, exist_ok=True)
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = [pool.submit(download_one, rec, dest_dir, label) for rec in records]
        for fut in as_completed(futures):
            done += 1
            row = fut.result()
            if row:
                rows.append(row)
            if done % 100 == 0:
                print(f"    {done}/{len(records)} processed ({label}) ...")
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    logs = []

    print("TARGET (positives):")
    logs += download_records(fetch_image_records(TARGET_SPECIES, MAX_TARGET_IMAGES), "target")

    # Split the negative quota across the look-alike species so the two classes
    # stay roughly balanced (imbalanced data quietly wrecks a beginner classifier).
    neg_quota = max(50, MAX_TARGET_IMAGES // len(NEGATIVE_SPECIES))
    print("\nNEGATIVES:")
    for sp in NEGATIVE_SPECIES:
        logs += download_records(fetch_image_records(sp, neg_quota), "not_target")

    log_path = os.path.join(OUT_DIR, "licenses.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filepath", "label", "species", "occurrence_id",
            "url", "license", "rights_holder", "creator"])
        writer.writeheader()
        writer.writerows(logs)

    n_target = sum(r["label"] == "target" for r in logs)
    n_neg = sum(r["label"] == "not_target" for r in logs)
    print(f"\nDone. {n_target} target + {n_neg} not_target images downloaded.")
    print(f"Attribution log written to {log_path}")


if __name__ == "__main__":
    main()
