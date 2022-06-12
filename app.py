import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import model_from_json

labels = {
    1: 'Atelectasis',
    2: 'Cardiomegaly',
    3: 'Consolidation',
    4: 'Edema',
    5: 'Effusion',
    6: 'Emphysema',
    7: 'Fibrosis',
    8: 'Hernia',
    9: 'Infiltration',
    10: 'Mass',
    11: 'Nodule',
    12: 'Pleural Thickening',
    13: 'Pneumonia',
    14: 'Pneumothorax'
    }

@st.cache(allow_output_mutation=True)
def load_cache_model(config_path, weights_path):

    # Load model config
    with open(config_path, mode='r') as f:
        arr = f.readlines()

    json_str = arr[0].replace('\\', '')[1:-1]

    new_model = model_from_json(json_str)

    # Load Weights
    new_model.load_weights(weights_path)

    return new_model


if __name__ == '__main__':

    # Load and cache model
    model = load_cache_model(Path(r'./model_files/model-8b-config.json'), Path(r'./model_files/model-8b-weights.h5'))

    # Streamlit GUI
    st.title('Demo - Lung Disease Diagnosis')
    st.write("""
    Developer: Tung Nguyen, www.github.com/tungtngyn

    This demo is a deep learning model capable of predicting 14 different lung diseases given a Chest X-Ray image.

    This model utilizes transfer learning, with a base Xception architecture pre-trained on ImageNet images. It is then further trained on [this dataset](https://www.kaggle.com/nih-chest-xrays/data).


    Download sample Chest X-Ray images from this [Github repo](https://github.com/tungtngyn/lung-disease-diagnosis-app/tree/main/imgs) and test it out here! The image names are their true labels.

    An example is pre-loaded below. The default classification threshold is set to 0.5, though this would be optimized based on domain knowledge if this were a real deployment.

    Note: This app looks better in light-mode! 
    """)

    file = st.file_uploader('Upload an image')

    if file:
        img = Image.open(file).convert('RGB')
        img_resized = img.resize((256, 256))
        st.title("Uploaded Image:")
        st.image(img_resized)

    else:
        img = Image.open(Path('./imgs/6_Effusion.png')).convert('RGB')
        img_resized = img.resize((256, 256))
        st.title("Sample Image:")
        st.write("""
        Sample Image: 6_Effusion.png
        """)
        st.image(img_resized)

    img_arr = img_to_array(img_resized)
    img_batch = np.expand_dims(img_arr, axis=0)
    img_processed = preprocess_input(img_batch)

    st.title("Model Predictions:")
    predictions = model.predict(img_processed)

    pred = {}
    for i, v in enumerate(predictions[0]):
        pred[labels[i+1]] = [float(v*100)]

    df = pd.DataFrame.from_dict(pred, orient='index', columns=['Probability (%)'])
    
    fig, ax = plt.subplots()
    y = np.arange(len(df.index))
    probs = df['Probability (%)'].values

    cc=list(map(lambda p: 'midnightblue' if p <= 50 else 'red', probs))
    p1 = ax.barh(y, probs, color=cc);

    ax.set_yticks(y);
    ax.set_yticklabels(df.index);
    ax.bar_label(p1, fmt='%.1f')
    ax.set_xlim(right=110)
    ax.set_title('Classification Summary')
    ax.set_xlabel('Probability (%)')
    ax.set_ylabel('Lung Disease')

    plt.axvline(x=50, color='k', linestyle='--')
    st.write(fig)

    st.write(df.round(decimals=1).to_html(escape=False), unsafe_allow_html=True)

    st.title('Additional Sample Images')
    st.write('Shown below are some additional samples of Chest X-Ray images and their associated true label.')
    img = Image.open(Path('./imgs/overview.png'))
    st.image(img)