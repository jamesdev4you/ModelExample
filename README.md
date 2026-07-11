hi team. 

You need to download miniconda from https://www.anaconda.com/docs/getting-started/installation installer.

Look for something like this as you go through the installation process: 
Miniconda3-latest-Linux-x86_64.sh

Except instead of Linux, do it for your OS (mac or windows). 

Whatever you download from that, make sure you place it inside of DataGrabber folder. 


# CLAUDE RECOMMENDATIONS BELOW

Running the Ibis Classifier & Honing the Model

A companion to the README. Picks up right after you've dropped the Miniconda
installer into the DataGrabber folder.

The whole project is three scripts run in order:

download_gbif_images.py   ->  grabs labeled photos from GBIF/iNaturalist
split_dataset.py          ->  sorts them into train / val / test
train_ibis.py             ->  trains the model and exports it for iOS

You run them once, in that order, and you get a trained model out the end. The
rest of this doc is about running them cleanly and then making the model better.


Part 1 — One-time setup

1a. Finish installing Miniconda

You've already downloaded the installer (Miniconda3-latest-<YourOS>...) into the
DataGrabber folder. Now install it:


Windows: double-click the .exe and click through. Then open the new
"Anaconda Prompt" from the Start menu — that's the terminal you'll use.
macOS / Linux: open a terminal, cd into DataGrabber, and run the
installer script:


bash  bash Miniconda3-latest-*.sh

Close and reopen the terminal when it finishes.

Check it worked — this should print a version number:

bashconda --version

1b. Make a clean environment

A conda "environment" is a sandbox so this project's packages don't collide with
anything else on your machine. Create one called ibis (the training script even
expects that name):

bashconda create -n ibis python=3.11 -y
conda activate ibis

You'll know it's active because your prompt now starts with (ibis). You have to
run conda activate ibis every time you open a new terminal for this project.

1c. Install the packages

bash# needed by the downloader
pip install requests

# needed by the trainer
pip install torch torchvision coremltools

Two notes:


GPU (optional, big speed-up): if you have an NVIDIA GPU, install the CUDA
build of PyTorch instead of the plain torch above. Grab the exact command
from the picker at https://pytorch.org/get-started/locally/. Without a GPU it
still works fine on CPU, just slower.
coremltools is the picky one. It's an Apple export tool and is only
properly supported on macOS and Linux. On Windows it may refuse to install
or fail at the very last "export to Core ML" step. See Troubleshooting below if
that bites you — everything except the final export still works on Windows.


split_dataset.py needs no packages — it's pure standard library.


Part 2 — Running the pipeline

Make sure you're in the project folder with (ibis) active, then go in order.

Step 1 — Download the images

bashpython download_gbif_images.py

What it does: queries GBIF for openly-licensed photos of the White Ibis
(positives) and a handful of look-alike birds (negatives — egrets, storks,
herons, etc.), downloads them, and writes an attribution log.

You'll end up with:

data/
  target/          <- White Ibis photos
  not_target/      <- the look-alike birds
  licenses.csv     <- credit for every photo (keep this!)

This is the slow step — it's pulling ~1,500 positives plus negatives over the
network. Expect it to run for a while and to skip some dead links (normal).


Keep data/licenses.csv. It's your proof that every photo was CC0 / CC-BY and
your list of photographers to credit. Don't delete it.



Step 2 — Split into train / val / test

bashpython split_dataset.py

This copies your images into:

data_split/
  train/{target,not_target}/   (70%)
  val/{target,not_target}/     (15%)
  test/{target,not_target}/    (15%)

The clever bit: it splits by observation, not by photo. iNaturalist sightings
often have several near-identical shots of the same bird from the same minute. If
copies of one sighting landed in both train and test, your test score would just
be measuring memorization — a lie that looks like success. This script keeps all
photos of one sighting together in the same split, so your test score is honest.

It copies by default (your originals in data/ are untouched).

Step 3 — Train and export

bashpython train_ibis.py

This trains the classifier in two phases (first the new head, then a gentle
fine-tune of the last few backbone layers), then prints an honest scorecard on
the held-out test set, then exports the model.

Watch the per-epoch lines as it runs:

[head] epoch 3/8   train_acc=0.880  val_acc=0.842  gap=+0.038


train_acc — how well it does on photos it's studying.
val_acc — how well it does on photos it's not studying. This is the number
that matters.
gap — train_acc − val_acc. A big positive gap = overfitting (memorizing
instead of learning). More on this below.


At the end you get:

ibis_model.pt                  <- PyTorch weights (for retraining / PyTorch use)
WildlifeClassifier.mlpackage   <- the Core ML model, ready to drop into an iOS app

That .mlpackage is a folder; inside it are the files you already have as samples
(weight.bin, Manifest.json, and a model spec). Normalization and softmax are
baked in, so the app just feeds it an RGB image and reads out the probabilities.


Part 3 — Reading the scorecard

At the end, train_ibis.py prints something like:

================ TEST RESULTS ================
  accuracy : 0.910  (182/200)
  precision: 0.930   (of badge PASSES, how many were really ibis)
  recall   : 0.880   (of real ibis, how many we caught)
  false positives (NOT-ibis wrongly passed): 7   <- badge cheating risk
  false negatives (real ibis wrongly failed): 17  <- annoyed-user risk
==============================================

For a "photograph the animal to earn a badge" app, the two error types cost you
different things, so decide which you care about more:


False positives = a non-ibis got waved through. That's someone earning the
badge with the wrong photo — cheating risk. If this matters most, chase
precision.
False negatives = a real ibis got rejected. That's a legit user getting told
"nope" — annoyance / churn risk. If this matters most, chase recall.


You usually can't max both at once; pick the one that fits your app and tune
toward it.


Part 4 — Honing the model

Improving a model is a loop, not a one-shot. Change one thing, retrain, look
at val_acc and the gap, keep it or revert. Change several things at once and you
won't know which one helped.

Rule 1: data beats knobs

Before touching any hyperparameter, more/better data almost always helps more than
tuning. In order of impact:


More positives. Raise MAX_TARGET_IMAGES in download_gbif_images.py
(it starts at 1500) and re-run the whole pipeline.
Harder negatives. Your negatives are the look-alikes the model keeps
confusing for an ibis. If it keeps passing egrets, get more egret photos. Edit
NEGATIVE_SPECIES to add whatever it's actually failing on.
Balance. Roughly equal counts of target vs not_target. The trainer does
weight the classes to compensate, but starting balanced is better. The download
script already splits the negative quota across the look-alike species for this
reason — keep an eye on the per-class counts it prints.
Realistic photos. Your test photos should look like what real users will
actually snap (phone quality, odd angles, bad light). A model that aces clean
iNaturalist shots can still flop on a blurry backyard photo.


Rule 2: read the gap, then pick the right knob

Look at the gap (train_acc − val_acc) on the last few epochs:

What you seeWhat it meansWhat to doBig gap (train high, val low, e.g. +0.15)Overfitting — memorizingMore data, or turn up regularization (below)Both low (train & val both mediocre)Underfitting — not learning enoughTrain longer, or turn regularization down, or unfreeze more layersBoth high, small gapYou're in good shapeBank it; only chase small gains from here

All the knobs live in the CONFIG block at the top of train_ibis.py:

If you're overfitting (big gap), turn these UP:


DROPOUT (currently 0.4) — randomly ignores neurons so it can't lean on any
one. Try 0.5.
WEIGHT_DECAY (currently 1e-4) — penalizes over-complex solutions. Try 1e-3.
Data augmentation already helps here (random crops/flips/color jitter in
train_tf) — adding more positives helps most of all.


If you're underfitting (both low), turn these UP:


UNFREEZE_LAST_N (currently 3) — how many backbone layers get fine-tuned.
More = more capacity to adapt. Try 4 or 5.
EPOCHS_HEAD / EPOCHS_FINETUNE — let it train longer. (Early stopping via
PATIENCE means it won't waste time if it plateaus.)


Learning rates — the touchiest knobs:


LR_HEAD (1e-3) trains the fresh head; can be relatively high.
LR_FINETUNE (1e-5) is deliberately tiny so fine-tuning doesn't blow away the
pretrained features. If fine-tuning makes things worse, this is too high —
lower it (5e-6). If fine-tuning does nothing, nudge it up (2e-5).


Other useful ones:


PATIENCE (5) — epochs to wait for improvement before stopping a phase.
Raise it to let training run longer before giving up.
BATCH_SIZE (32) — lower it (16) if you hit out-of-memory errors on GPU.
IMG_SIZE (224) — bigger can help accuracy but is slower and memory-hungry;
224 is the standard sweet spot for this backbone. Leave it unless you have a
reason.


A sane tuning recipe


Run once with the defaults. Note val_acc and the gap — that's your baseline.
If overfitting: add data first; if you can't, bump DROPOUT to 0.5 or
WEIGHT_DECAY to 1e-3 (one at a time).
If underfitting: bump UNFREEZE_LAST_N to 4, or raise the epoch caps.
Retrain, compare to baseline, keep the change only if val_acc went up.
Repeat, one knob at a time.


Whenever you change the actual images in data/, re-run split_dataset.py before
retraining so the new photos make it into the splits.


Part 5 — Pointing it at a different animal

The pipeline isn't ibis-specific. To retarget it, edit the config block at the top
of download_gbif_images.py:


TARGET_SPECIES — the scientific name of your new animal (e.g.
"Sciurus carolinensis" for an Eastern Gray Squirrel). Use the scientific
name; GBIF matches those reliably.
NEGATIVE_SPECIES — the look-alikes / co-habitants a user might photograph by
mistake instead. Good negatives are the whole game: pick things that are
genuinely easy to confuse with your target.
COUNTRY — restrict to a region, or set to None for worldwide. (Positives are
worldwide by design — the animal looks the same everywhere. Region mainly
matters for choosing realistic negatives.)


Then re-run all three steps in order. The rest of the pipeline just follows the
target / not_target folders, so no other changes are needed. (You may want to
rename OUT_WEIGHTS / OUT_COREML in train_ibis.py so the output isn't still
called "ibis".)


Part 6 — Troubleshooting

Windows/macOS: crash with "An attempt has been made to start a new process
before the current process has finished its bootstrapping phase" (or a
multiprocessing error) at training start.
This is the DataLoader's worker processes (num_workers=4) on non-Linux systems.
Quickest fix: open train_ibis.py and set every num_workers=4 to num_workers=0.
Slower, but reliable on any OS.

coremltools won't install / the final export step errors on Windows.
coremltools targets macOS and Linux. Options: (a) run the whole thing on a Mac or
Linux box; (b) use WSL (Windows Subsystem for Linux) on your Windows machine;
or (c) do the training on Windows and just run the final Core ML export on a
Mac/Linux machine using the saved ibis_model.pt. The training itself doesn't
need coremltools at all.

Download is slow or lots of images "skip."
Normal — some GBIF image links are dead or time out, and the script just skips
them. If you want more images, raise MAX_TARGET_IMAGES and re-run; already-
downloaded files are cached and won't re-download.

Classes look imbalanced (e.g. way more negatives than positives).
Check the per-class counts the trainer prints. Either download more of the smaller
class, or accept it — the trainer already weights the loss to compensate — but
balanced data is still better.

val_acc bounces around a lot between epochs.
Usually a too-small validation set or too-high learning rate. Get more data, or
lower LR_FINETUNE.


