import json
import matplotlib.pyplot as plt

import streamlit as st

from persim import plot_diagrams, PersistenceImager


@st.cache(
    show_spinner=False,
)
def load_class_labels_dicts():
    folders_to_numbers = json.load(open("data/classmap.json", "rb"))
    numbers_to_names = json.load(open("data/ImageNet_labels.json", "rb"))
    names_to_numbers = {v: k for k, v in numbers_to_names.items()}
    numbers_to_folders = {v: k for k, v in folders_to_numbers.items()}
    return names_to_numbers, numbers_to_folders
