from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import streamlit as st
import nltk

MODEL_NAME_AMAZON = "LiYuan/amazon-review-sentiment-analysis"
MODEL_NAME_TWITTER = "cardiffnlp/twitter-roberta-base-sentiment-latest"

AMAZON_ROW_LABELS = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
TWITTER_ROW_LABELS = ['Negative', 'Neutral', 'Positive']

AMAZON_ROW_VALUES = list(range(1,6))
TWITTER_ROW_VALUES = [0, 0.5, 1]


@st.cache_resource
def download_model(model_name: str = MODEL_NAME_AMAZON):
    '''
    Downloads the model from huggingface.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    return nlp

@st.cache_resource
def init():
    '''
    Downloads the stopwords for nltk.
    '''
    nltk.download('stopwords')
