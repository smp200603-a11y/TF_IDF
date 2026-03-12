import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.title("Buscador de similitud con TF-IDF")

st.write("""
Cada línea ingresada se interpreta como un **documento independiente**.  
Puede ser una oración corta o incluso un pequeño párrafo.

El sistema utiliza **TF-IDF y similitud del coseno** para encontrar el texto
que más se parece a la pregunta escrita por el usuario.

El procesamiento aplica **normalización y stemming**, por lo que palabras
como *running* y *run* pueden ser tratadas como equivalentes.
""")

# Texto inicial de ejemplo
docs_input = st.text_area(
    "Ingresa varios textos (uno por línea):",
    "Dogs bark when they see strangers.\nCats like to sleep during the day.\nDogs and cats sometimes live together."
)

question = st.text_input("Escribe una pregunta para buscar coincidencias:", "Who sleeps during the day?")

# Inicializar stemmer
stemmer = SnowballStemmer("english")

def limpiar_y_stem(texto: str):
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    tokens = [t for t in texto.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

if st.button("Analizar textos"):
    documentos = [d.strip() for d in docs_input.split("\n") if d.strip()]

    if len(documentos) == 0:
        st.warning("Debes ingresar al menos un texto.")
    else:

        vectorizador = TfidfVectorizer(
            tokenizer=limpiar_y_stem,
            stop_words="english",
            token_pattern=None
        )

        matriz = vectorizador.fit_transform(documentos)

        df = pd.DataFrame(
            matriz.toarray(),
            columns=vectorizador.get_feature_names_out(),
            index=[f"Texto {i+1}" for i in range(len(documentos))]
        )

        st.write("### Matriz TF-IDF generada")
        st.dataframe(df.round(3))

        # Transformar pregunta
        pregunta_vec = vectorizador.transform([question])

        similitudes = cosine_similarity(pregunta_vec, matriz).flatten()

        mejor = similitudes.argmax()
        mejor_texto = documentos[mejor]
        mejor_valor = similitudes[mejor]

        st.write("### Resultado de búsqueda")
        st.write(f"**Pregunta ingresada:** {question}")
        st.write(f"**Texto más similar (Texto {mejor+1}):** {mejor_texto}")
        st.write(f"**Nivel de similitud:** {mejor_valor:.3f}")

        tabla_sim = pd.DataFrame({
            "Texto": [f"Texto {i+1}" for i in range(len(documentos))],
            "Contenido": documentos,
            "Similitud": similitudes
        })

        st.write("### Comparación de similitudes")
        st.dataframe(tabla_sim.sort_values("Similitud", ascending=False))

        vocabulario = vectorizador.get_feature_names_out()
        stems_pregunta = limpiar_y_stem(question)

        coincidencias = [s for s in stems_pregunta if s in vocabulario and df.iloc[mejor].get(s, 0) > 0]

        st.write("### Palabras clave de la pregunta encontradas en el texto:", coincidencias)
