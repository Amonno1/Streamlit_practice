
import streamlit as st
import pandas as pd
import numpy as np

st.write("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
sample = st.text_input("Nhap review")
st.write("ssad", sample)

reviews = [
    'A very good film of Tony Stark',
    'Fantastic fighting scene , I love watching Jet Li',
    'Not bad ! I was impressed by the action scene',
    'The love story was amazing , but I do not prefer romantic movie at all',
    'The main actor was stupid',
    'I can not imagine such an ugggly guy like the main actor'
]
labels = [
    'Positive',
    'Positive',
    'Positive',
    'Negative',
    'Negative',
    'Negative'
]

# Rule: "Good", "Fantastic", "Amazing" - "Bad", "Ugly", "Stupid"
positive_words = ["good", "fantastic", "amazing"]
negative_words = ["bad", "ugly", "stupid"]
def review_classification(review):
  # lowercase all the words in the review
  review_lower = review.lower()
  
  # split the text to separated words
  review_words = review_lower.split()

  # Assign label based on the keywords
  for word in review_words:
    if word in positive_words:
      return 'Positive'
    if word in negative_words:
      return 'Negative'
  
  # return neutral by default
  return 'Neutral'

all_tokens = [] # we use all_tokens variable to store all the words in all the reviews
for review in reviews:
  review_lower = review.lower()
  tokens = review_lower.split()
  all_tokens = all_tokens + tokens
vocab = set(all_tokens)

def review_to_vector(review):
  review_vector = []
  review_lower = review.lower()
  review_tokens = review_lower.split()
  for token in vocab:
    if token in review_tokens:
      review_vector.append(1)
    else:
      review_vector.append(0)
  return review_vector

reviews2vectors = [review_to_vector(r) for r in reviews]
st.write(reviews[0])
st.write(len(reviews2vectors))
st.write(reviews2vectors[0])
