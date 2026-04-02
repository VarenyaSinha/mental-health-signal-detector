# Ethical Considerations

## Dataset
This project uses publicly available Reddit posts from mental health subreddits.
All data was collected via the HuggingFace Datasets library from the
`solomonk/reddit_mental_health_posts` dataset. No personally identifiable
information was used or stored.

## Intended Use
This tool is intended strictly for research and educational demonstration of
NLP classification techniques. It is **not** a clinical or diagnostic tool.

## Known Limitations
- The model was trained on Reddit data and may not generalise to other platforms
  or clinical language.
- Classes like depression and aspergers share overlapping language, leading to
  clinically adjacent misclassifications (F1: 0.76–0.77 for these classes).
- The model cannot account for sarcasm, humour, or cultural context.
- Subreddit labels are used as proxies for mental health categories — they
  reflect community identity, not clinical diagnosis.

## Misuse Warning
This model should never be used to make decisions about real individuals'
mental health. Deploying this in any clinical or screening context without
expert oversight would be irresponsible and potentially harmful.

## Responsible AI
Class imbalance was handled via inverse-frequency class weights during training.
Model confidence scores are surfaced in the UI to communicate uncertainty.
A disclaimer is shown on every prediction.