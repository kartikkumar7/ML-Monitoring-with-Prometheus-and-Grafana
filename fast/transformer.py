from transformers import pipeline

summarizer = pipeline(task="summarization", model="sshleifer/distilbart-cnn-6-6", tokenizer="sshleifer/distilbart-cnn-6-6",framework="pt")

def summarize(text):
    summary = summarizer(text, max_length=40, min_length=20, do_sample=False)
    return summary