import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


def load_model():
    try:
        model = tf.keras.models.load_model('saved_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise SystemExit()



def load_properties():
    try:
        with open('saved_objects.pkl', 'rb') as file:
            aa_properties = pickle.load(file)
        return aa_properties['properties_dict']
    except Exception as e:
        st.error(f"Error loading properties: {e}")
        raise SystemExit()

def encode_with_properties(prot_primary_seq, properties_dict):
    encoded_seqs = []
    for sequence in prot_primary_seq:
        encoded_seq = []
        for aa in sequence:
            if aa in properties_dict:
                encoded_seq.append(properties_dict[aa])
            else:
                encoded_seq.append([0.0] * len(next(iter(properties_dict.values()))))  # Default to zeros if unknown
        encoded_seqs.append(encoded_seq)
    return encoded_seqs

st.title("Protein Secondary Structure Prediction")

sequence = st.text_input("Enter protein sequence:")

model = load_model()
properties_dict = load_properties()

if st.button("Predict"):
    if sequence:
        
        encoded_sequence = encode_with_properties([sequence], properties_dict)
        max_length = len(sequence)  
        padded_sequence = pad_sequences(encoded_sequence, maxlen=max_length, padding='post', dtype='float32')

        
        padded_sequence = np.array(padded_sequence)
        if len(padded_sequence.shape) == 2:
            padded_sequence = np.expand_dims(padded_sequence, axis=0)

       
        predicted_structure = model.predict(padded_sequence)
        classes = ['H', 'B', 'T']
        predicted_classes = np.argmax(predicted_structure, axis=-1)
        predicted_secondary_structure = [classes[pred] for pred in predicted_classes[0]]

        st.write("Predicted Secondary Structure:", "".join(predicted_secondary_structure))
    else:
        st.write("Please enter a valid protein sequence.")
