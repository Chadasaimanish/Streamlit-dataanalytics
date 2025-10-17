import streamlit as st
import spacy
from spacy import displacy
import pandas as pd

# --- Load spaCy Model ---
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
st.title("spaCy NER + POS Tagging")

# --- Input Widget ---
input_text = st.text_area(
    "Enter text for analysis:",
    key='text_area',
    height=200,
    value="Apple is looking at buying U.K. startup for $1 billion."
)

text = st.session_state.get('text_area', '')

if text.strip():
    doc = nlp(text)

    # --- Display Named Entities ---
    html = displacy.render(doc, style="ent", jupyter=False)
    st.write("*Detected Named Entities:*", unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)

    # --- Optional: Show NER table ---
    ner_entities = [(ent.text, ent.label_) for ent in doc.ents]
    if ner_entities:
        st.markdown("*Entity Table:*")
        st.table(ner_entities)
    else:
        st.info("No named entities found in the provided text.")

    # --- Show POS for all tokens ---
    token_data = [(token.text, token.pos_, token.dep_) for token in doc]
    st.markdown("*POS Tagging of All Words:*")
    st.table(pd.DataFrame(token_data, columns=["Token", "POS", "Dependency"]))

else:
    st.info("Paste or select some text to see results.")
