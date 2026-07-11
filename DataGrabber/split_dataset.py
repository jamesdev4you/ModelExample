#!/usr/bin/env python3
"""
Split the culled images into train / val / test sets -- BY OBSERVATION.

Why by observation: many GBIF / iNaturalist sightings have several near-identical
photos (same bird, same minute). If copies of one sighting land in both the
training and test sets, your test score measures memorization, not skill -- it
looks great and is a lie. This script keeps every photo from a single
observation together in the same split.

It reads the occurrence id that download_gbif_images.py baked into each filename
(<species>_<occurrenceid>_<hash>.jpg) and groups on that.

Run (no extra packages needed -- pure standard library):
    python split_dataset.py

Produces:
    data_split/
      train/{target,not_target}/
      val/{target,not_target}/
      test/{target,not_target}/
"""

import os
import random
import shutil
from collections import defaultdict

# ----------------------------- CONFIG -----------------------------
SRC_DIR = "data"                 # folder holding target/ and not_target/
OUT_DIR = "data_split"           # where the split copies are written
CLASSES = ["target", "not_target"]
RATIOS = (0.70, 0.15, 0.15)      # train, val, test  (must sum to 1.0)
SEED = 42                        # fixed -> identical split every run (reproducible)
COPY = True                      # True = copy (keeps your originals); False = move
# ------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def occurrence_id(filename):
    """Pull the occurrence id from '<species>_<occid>_<hash>.ext'.

    The occurrence id is always the second-to-last underscore token, so this
    works no matter how many words are in the species name.
    """
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    return parts[-2] if len(parts) >= 2 else stem


def split_observations(obs_ids):
    """Shuffle observation ids (deterministically) and cut into train/val/test."""
    obs_ids = sorted(obs_ids)                 # sort first so the shuffle is reproducible
    random.Random(SEED).shuffle(obs_ids)
    n = len(obs_ids)
    n_train = int(n * RATIOS[0])
    n_val = int(n * RATIOS[1])
    return {
        "train": set(obs_ids[:n_train]),
        "val":   set(obs_ids[n_train:n_train + n_val]),
        "test":  set(obs_ids[n_train + n_val:]),
    }


def main():
    assert abs(sum(RATIOS) - 1.0) < 1e-6, "RATIOS must sum to 1.0"
    transfer = shutil.copy2 if COPY else shutil.move
    summary = defaultdict(lambda: defaultdict(int))

    for cls in CLASSES:
        src = os.path.join(SRC_DIR, cls)
        if not os.path.isdir(src):
            print(f"!! missing folder: {src} -- skipping")
            continue

        # group this class's images by observation id
        groups = defaultdict(list)
        for fname in os.listdir(src):
            if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                groups[occurrence_id(fname)].append(fname)

        # split the OBSERVATIONS (not the photos), per class, to keep balance
        assignment = split_observations(list(groups.keys()))

        for split, obs_set in assignment.items():
            dest = os.path.join(OUT_DIR, split, cls)
            os.makedirs(dest, exist_ok=True)
            for obs in obs_set:
                for fname in groups[obs]:
                    transfer(os.path.join(src, fname), os.path.join(dest, fname))
                    summary[split][cls] += 1

        n_obs = len(groups)
        n_imgs = sum(len(v) for v in groups.values())
        print(f"{cls}: {n_imgs} images across {n_obs} observations")

    print("\nSplit complete (image counts):")
    for split in ["train", "val", "test"]:
        line = "   ".join(f"{cls}={summary[split][cls]}" for cls in CLASSES)
        print(f"  {split:5s}  {line}")
    print(f"\nOutput written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()