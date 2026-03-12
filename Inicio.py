import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer
from PIL import Image

st.title("Buscador de textos con TF-IDF")

# Imagen
imagen = Image.open("pollo.png")
st.image(imagen, width=250)

st.write("Escribe varios textos y luego una pregunta. El sistema buscará el texto más parecido.")

# Textos de ejemplo
text_input = st.text_area(
    "Textos (uno por línea):",
    "Dogs bark loudly.\nCats sleep during the day.\nDogs and cats can live together."
)

question = st.text_input("Pregunta:", "Who sleeps during the day?")

stemmer = SnowballStemmer("english")

def procesar(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    palabras = [t for t in texto.split() if len(t) > 1]
    return [stemmer.stem(p) for p in palabras]

if st.button("Buscar coincidencia"):

    docs = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(docs) == 0:
        st.warning("Ingresa al menos un texto.")
    else:

        vectorizer = TfidfVectorizer(
            tokenizer=procesar,
            stop_words="english",
            token_pattern=None
        )

        X = vectorizer.fit_transform(docs)

        df = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(docs))]
        )

        st.subheader("Matriz TF-IDF")
        st.dataframe(df.round(3))

        q_vec = vectorizer.transform([question])

        sims = cosine_similarity(q_vec, X).flatten()

        best = sims.argmax()

        st.subheader("Resultado")
        st.write("Pregunta:", question)
        st.write("Documento más cercano:", docs[best])
        st.write("Similitud:", round(sims[best],3))

        tabla = pd.DataFrame({
            "Documento":[f"Doc {i+1}" for i in range(len(docs))],
            "Texto":docs,
            "Similitud":sims
        })

        st.subheader("Comparación")
        st.dataframe(tabla.sort_values("Similitud", ascending=False))
