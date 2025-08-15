# Fake News Detector (Fun Project)

This is a small side project where I use a CLIP-based model to classify news as **real** or **fake** based on both an image and its caption.

## What's inside
- `code.ipynb` → the main notebook with all training and evaluation code plus a testing cell where you can hqve qn interface to insert new image and caption for testing.
- `checkpoints/` → model weights (ignored on GitHub).
- `submission.csv` → example submission for the test set.

## How it works
1. Loads an image and its caption.
2. Encodes both using CLIP.
3. Combines features and passes them through a classifier head.
4. Predicts "real" or "fake" news.

## Running it
Just open `code.ipynb` in Jupyter Notebook and run the cells.
