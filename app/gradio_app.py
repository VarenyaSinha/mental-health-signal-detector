import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gradio as gr
from transformers import DistilBertTokenizer
from src.model import build_model
from src.dataset import ID2LABEL, LABEL2ID

# ── Config ────────────────────────────────────────────────────────────────────
MAX_LENGTH = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "outputs/models/best_model.pt")

LABEL_DESCRIPTIONS = {
    "ADHD":       "Posts suggest themes common in ADHD communities — focus, distraction, impulsivity, productivity struggles.",
    "OCD":        "Posts suggest themes common in OCD communities — intrusive thoughts, compulsions, rituals, anxiety loops.",
    "aspergers":  "Posts suggest themes common in Aspergers/autism communities — social navigation, sensory experience, identity.",
    "depression": "Posts suggest themes common in depression communities — low mood, hopelessness, isolation, fatigue.",
    "ptsd":       "Posts suggest themes common in PTSD communities — trauma responses, hypervigilance, intrusive memories.",
}

DISCLAIMER = """
⚠️ **Important:** This tool is for research and educational purposes only.
It is not a diagnostic tool and should never replace professional mental health assessment.
If you or someone you know is struggling, please reach out to a qualified mental health professional.
"""

# ── Load model once at startup ────────────────────────────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded ✓")


# ── Inference function ────────────────────────────────────────────────────────
def predict(text: str):
    if not text or len(text.strip()) < 10:
        return {}, "Please enter at least a sentence.", ""

    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).squeeze(0)

    # Build confidence dict for Gradio label component
    scores = {ID2LABEL[i]: float(probs[i]) for i in range(len(ID2LABEL))}

    # Top prediction
    top_label = max(scores, key=scores.get)
    top_conf  = scores[top_label]
    description = LABEL_DESCRIPTIONS[top_label]

    # Confidence note
    if top_conf > 0.80:
        conf_note = "High confidence"
    elif top_conf > 0.55:
        conf_note = "Moderate confidence"
    else:
        conf_note = "Low confidence — text may span multiple categories"

    summary = f"**Predicted category: {top_label}** ({top_conf:.1%} — {conf_note})\n\n{description}"

    return scores, summary, DISCLAIMER


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Mental Health Signal Detector") as demo:

    gr.Markdown("# 🧠 Mental Health Signal Detector")
    gr.Markdown(
        "Enter a social media post or text excerpt. The model will classify it into one of five "
        "mental health community categories using a fine-tuned DistilBERT model trained on Reddit data."
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Input text",
                placeholder="Paste a Reddit post or any text here...",
                lines=6
            )
            submit_btn  = gr.Button("Analyse", variant="primary")
            clear_btn   = gr.Button("Clear")

        with gr.Column(scale=2):
            label_output = gr.Label(
                label="Confidence scores per category",
                num_top_classes=5
            )
            summary_output    = gr.Markdown(label="Prediction summary")
            disclaimer_output = gr.Markdown(label="")

    # Example inputs
    gr.Examples(
        examples=[
            ["I can't stop checking if I locked the door. I've gone back 7 times already and I still don't feel sure."],
            ["I was diagnosed last year and suddenly everything makes sense. Why I couldn't sit still in class, why I lose things constantly."],
            ["Some days I just can't get out of bed. Everything feels pointless and I don't know how to explain it to anyone."],
            ["Social situations drain me completely. I need days to recover after any kind of gathering."],
            ["Loud noises still make me jump. My heart races and I'm back there again even though it was years ago."],
        ],
        inputs=text_input
    )

    submit_btn.click(
        fn=predict,
        inputs=text_input,
        outputs=[label_output, summary_output, disclaimer_output]
    )
    clear_btn.click(lambda: ("", {}, "", ""), outputs=[text_input, label_output, summary_output, disclaimer_output])

if __name__ == "__main__":
    demo.launch()