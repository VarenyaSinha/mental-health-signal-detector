import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Clean a single Reddit post.
    Removes URLs, Reddit-specific artifacts, excess whitespace.
    Keeps the raw emotional language intact — do NOT remove stopwords
    or stem/lemmatize. BERT was trained on natural text and needs it.
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove Reddit mentions like u/username and r/subreddit
    text = re.sub(r'u/\w+|r/\w+', '', text)

    # Remove [deleted] and [removed] placeholders
    text = re.sub(r'\[deleted\]|\[removed\]', '', text)

    # Remove special characters but keep sentence punctuation
    # Keeping .,!?- because they carry emotional signal
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)

    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['text'] = df['text'].apply(clean_text)

    # Drop anything that became too short after cleaning
    df = df[df['text'].str.split().str.len() >= 5].reset_index(drop=True)

    return df


if __name__ == "__main__":
    import os

    for split in ['train', 'val', 'test']:
        path = f"data/processed/{split}.csv"
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        df = load_and_clean(path)
        out = f"data/processed/{split}_clean.csv"
        df.to_csv(out, index=False)
        print(f"{split}: {len(df)} rows saved → {out}")