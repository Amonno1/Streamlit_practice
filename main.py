
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

for review in reviews:
  st.write(review_classification(review))
