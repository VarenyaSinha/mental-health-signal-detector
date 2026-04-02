import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import DistilBertTokenizer

TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Paste your label mapping output from Cell 7 here
LABEL2ID = {
    'ADHD':       0,
    'OCD':        1,
    'aspergers':  2,
    'depression': 3,
    'ptsd':       4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


class RedditMentalHealthDataset(Dataset):
    def __init__(self, csv_path: str, max_length: int = 128):
        """
        max_length=256 not 512 — most posts are short, this halves
        memory usage and speeds up training with negligible accuracy loss.
        We verified only 3% of posts exceed 512 words anyway.
        """
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        label = int(row['label_id'])

        encoding = TOKENIZER(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(label, dtype=torch.long)
        }


if __name__ == "__main__":
    # Quick sanity check
    ds = RedditMentalHealthDataset('data/processed/train_clean.csv')
    sample = ds[0]
    print("input_ids shape:", sample['input_ids'].shape)
    print("attention_mask shape:", sample['attention_mask'].shape)
    print("label:", sample['label'].item(), "→", ID2LABEL[sample['label'].item()])
    print(f"\nDataset size: {len(ds)}")