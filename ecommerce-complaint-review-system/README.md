# AI-Based E-Commerce Complaint Review System

An end-to-end pipeline that automatically classifies customer complaints,
verifies whether any attached photo has been digitally tampered with, and
recommends a resolution (**Refund / Replacement / Escalation**) — combining
a **CNN + RNN + LSTM ensemble** for text classification, a separate
**image-authenticity CNN** for photo forensics, and an **LLM Agent** for
policy-aware reasoning. The whole thing runs in an elegant **Streamlit** app.

## Architecture

```
Customer Complaint (text + optional photo)
          |
     Preprocessing
          |
     CNN / RNN / LSTM  ---------------->  Image Authenticity CNN (ELA-based)
          |                                        |
      Model Fusion                          Real / Fake + confidence
          |                                        |
  Category + Severity + Confidence  <---------------
          |
       LLM Agent  (retrieve policy -> reason, factoring in image verdict)
          |
   Final Recommendation (Refund / Replacement / Escalation)
```

### Why three text models?
- **CNN** — picks up local n-gram patterns ("cracked screen", "still not delivered").
- **RNN** — captures short sequential/order dependencies.
- **LSTM** — captures longer-range dependencies and contrastive phrasing
  ("was fine, but then completely stopped working").

Their softmax outputs are fused via weighted averaging ("soft voting") in
`src/models/fusion.py`, producing a single category, severity, and confidence
score — more robust than relying on any one architecture.

### Image authenticity CNN — what it does and doesn't do
This CNN has exactly **one job**: deciding whether a photo a customer attaches
to a complaint (e.g. "here's my damaged product") shows signs of digital
tampering. It is completely separate from the text ensemble and is never used
for complaint classification.

It works via **Error Level Analysis (ELA)**, a standard image-forensics
technique:
1. The uploaded photo is re-saved at a known JPEG quality and diffed against
   the original. Untouched regions settle to a low, even error level;
   pasted-in, redrawn, or locally re-edited regions were compressed a
   different number of times and light up in the diff.
2. That ELA map (not the raw photo) is fed into a small CNN
   (`src/models/image_cnn.py`) trained to classify it as **Real** or **Fake**.
3. Training data is generated automatically: `src/manipulate.py` takes
   authentic photos and produces tampered copies via copy-move forgery,
   splicing, localized noise, and re-compression — so you don't need a
   pre-existing labeled fake-image dataset to get started.

If a photo is flagged as likely manipulated, the pipeline **always escalates
the case for manual fraud review**, regardless of what the text classifier says.

### Why an LLM Agent on top?
The neural models are good at *pattern recognition* (what kind of complaint
is this, how severe does it sound, does this photo look tampered) but have no
notion of business policy. The LLM Agent step:
1. **Retrieves** the relevant policy text for the predicted category from
   `policies/company_policy.json`.
2. **Reasons** over the complaint + classification + image verdict + policy
   using Claude, producing a structured recommendation with an explanation.
3. Falls back to a transparent **rule-based recommender** if no
   `ANTHROPIC_API_KEY` is set (so the whole system still runs offline/for
   demo purposes) — and always escalates to a human when classifier
   confidence is low or the photo looks fake.

## Project Structure

```
ecommerce-complaint-review-system/
├── streamlit_app.py             # Main Streamlit UI
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   ├── sample_complaints.csv     # ~60 labeled synthetic complaints
│   └── images/
│       └── real/                  # Authentic photos (source for training)
├── policies/
│   └── company_policy.json        # Per-category policy + default actions
├── scripts/
│   └── generate_sample_images.py  # Generates synthetic demo product photos
├── checkpoints/                    # Created after training (weights, vocab, label maps)
└── src/
    ├── preprocessing.py            # Text cleaning, vocabulary, label maps
    ├── dataset.py                  # PyTorch Dataset for text
    ├── image_utils.py               # Error Level Analysis (ELA) preprocessing
    ├── manipulate.py                 # Synthetic forgery generator (for training data)
    ├── models/
    │   ├── cnn_model.py              # TextCNN (category + severity heads)
    │   ├── rnn_model.py              # TextRNN (category + severity heads)
    │   ├── lstm_model.py             # TextLSTM (category + severity heads)
    │   ├── fusion.py                 # Weighted soft-voting fusion (text)
    │   └── image_cnn.py               # ImageAuthenticityCNN (Real vs Fake)
    ├── train.py                       # Trains the text CNN/RNN/LSTM ensemble
    ├── train_image_cnn.py              # Trains the image authenticity CNN
    ├── predict.py                       # Loads text checkpoints, runs ensemble inference
    ├── image_predict.py                  # Loads image CNN checkpoint, runs authenticity check
    ├── llm_agent.py                       # Policy retrieval + LLM/rule-based reasoning
    └── pipeline.py                         # Orchestrates the full flow end-to-end
```

## Setup

```bash
git clone <your-repo-url>
cd ecommerce-complaint-review-system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

To enable real LLM reasoning (recommended), set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"     # Windows: set ANTHROPIC_API_KEY=your-key-here
```

Without a key, the LLM Agent step automatically falls back to a deterministic,
policy-driven rule-based recommender — the pipeline still works fully offline.

## Usage

### 1. Train the text classification ensemble

```bash
python src/train.py --epochs 15
```

This builds the vocabulary, trains the CNN/RNN/LSTM ensemble on
`data/sample_complaints.csv`, and saves everything to `checkpoints/`.

> The bundled dataset is intentionally small (~60 rows) for demo purposes.
> Swap in a larger, real complaint dataset (same CSV schema: `text,category,severity`)
> for meaningfully higher accuracy.

### 2. Train the image authenticity CNN (optional but recommended)

```bash
python scripts/generate_sample_images.py --count 25   # only needed if you have no real photos yet
python src/train_image_cnn.py --epochs 15
```

`data/images/real/` ships empty — the generator script fills it with
synthetic procedurally-drawn "product photo" placeholders so the pipeline
is runnable out of the box. **Replace them with real, verified-authentic
customer/product photos before using this for anything beyond a demo.**

### 3. Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open the printed local URL. Type or paste a complaint, optionally attach a
photo, and click **Analyze Complaint** to see:
- The CNN/RNN/LSTM ensemble's category + severity with confidence bars
- The photo's ELA map and Real/Fake verdict (if a photo was attached)
- The final Refund/Replacement/Escalation recommendation with reasoning

### 4. Or run the pipeline from the command line

```bash
python -m src.pipeline "My package is three weeks late and support won't respond."
```

## Extending this project

- **Bigger dataset**: replace `data/sample_complaints.csv` and
  `data/images/real/` with real, larger labeled datasets for
  production-quality accuracy.
- **Pretrained embeddings**: swap the `nn.Embedding` layers for GloVe/Word2Vec
  or a transformer encoder for stronger text representations.
- **Learned fusion weights**: replace the fixed `(1/3, 1/3, 1/3)` fusion
  weights in `fusion.py` with weights tuned on a validation set, or a small
  learned meta-classifier ("stacking").
