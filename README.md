# Multimodal Fake News Detection with CLIP

Fake news often pairs a real-looking image with a misleading caption, so judging
either one alone misses the signal. This project classifies a news post as
**real** or **fake** from its **image and caption together**, by fine-tuning
OpenAI's CLIP with a small classification head.

## Results

| Split | Accuracy | Macro-F1 |
|---|---|---|
| Validation (2,500 posts, balanced) | **0.958** | **0.958** |

| | Precision | Recall |
|---|---|---|
| Fake | 0.93 | 0.99 |
| Real | 0.99 | 0.92 |

The model almost never lets a fake post through (99% recall on fake), at the cost of
flagging about 8% of real posts as fake. The test split is unlabeled, so predictions
on it are written to `submission.csv` instead of being scored.

## How it works

1. The caption is tokenised and the image is resized and normalised by `CLIPProcessor`.
2. CLIP (`openai/clip-vit-base-patch32`) produces a text embedding and an image embedding in a shared space.
3. The two embeddings are averaged (late fusion).
4. A head (LayerNorm, dropout 0.2, linear layer) predicts real vs fake.
5. CLIP and the head are fine-tuned end to end: AdamW (lr 1e-5), batch size 16,
   5 epochs, cross-entropy with label smoothing 0.1.

Training saves a resumable checkpoint every 20 steps, so runs can be interrupted
and continued without losing progress.

## Try it

The last notebook cells include a `predict_fake_real(image_path, caption)` function
and a **Gradio** interface: upload an image, type a caption, get a prediction.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook code.ipynb
```

Put the dataset's parquet files (`train-00000-of-00001.parquet`,
`validation-00000-of-00001.parquet`, `test-*.parquet`) in the project folder. Each
row holds the image bytes (`image`), the caption (`text`) and the label (`label`:
0 = fake, 1 = real). Checkpoints are written to `checkpoints/` (not tracked by git).

## Limitations and next steps

- Averaging the embeddings is the simplest fusion. Concatenation or cross-attention
  between image and text tokens could capture image-caption *mismatch* better.
- The checkpoint was selected on the same validation set that is reported; a
  separate labeled test set would give a cleaner estimate.
- Next: error analysis on misclassified posts, and checking whether the model relies
  on the caption, the image, or both (by evaluating with each one masked).

## Author

Rayen Latrech · [GitHub](https://github.com/rayenlatrech)
