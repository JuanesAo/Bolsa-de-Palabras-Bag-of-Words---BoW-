import streamlit as st
import pandas as pd
from typing import List
from transformers import pipeline

# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(
    page_title="Zero-Shot Text Classifier",
    page_icon="🧭",
)

st.title("🧭 Clasificador de Tópicos Zero-Shot (BART-MNLI)")
st.caption("Clasifica texto en etiquetas que el modelo nunca vio al entrenar, vía NLI.")

# -----------------------------
# Carga eficiente del modelo
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_zero_shot_pipeline():
    # device='cpu' para compatibilidad general; Streamlit Cloud no siempre tiene GPU.
    clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,           # CPU
    )
    return clf

clf = load_zero_shot_pipeline()

# -----------------------------
# Utilidades
# -----------------------------
def parse_labels(raw: str) -> List[str]:
    # Divide por coma o salto de línea, limpia espacios y evita vacíos
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    return [p for p in parts if p]

def to_df(labels: List[str], scores: List[float]) -> pd.DataFrame:
    return pd.DataFrame({"Etiqueta": labels, "Puntaje": scores})

# -----------------------------
# Sidebar de opciones
# -----------------------------
st.sidebar.header("Opciones")
multi_label = st.sidebar.toggle("Multi-label (permitir varias etiquetas verdaderas)", value=True)
hypothesis_template = st.sidebar.text_input(
    "Template de hipótesis (NLI)",
    value="Este texto trata sobre {}.",
    help="El modelo evalúa si el texto implica esta hipótesis. Ej: 'Este texto trata sobre {}.'"
)
sort_desc = st.sidebar.toggle("Ordenar desc por puntaje", value=True)

# -----------------------------
# Entradas principales
# -----------------------------
default_text = "Lionel Messi ganó otro título con la selección argentina y fue elegido mejor jugador del torneo."
default_labels = "deportes, política, economía, farándula, tecnología"

texto = st.text_area("📝 Texto a clasificar", value=default_text, height=160)
raw_labels = st.text_area("🏷️ Etiquetas (separadas por comas o nuevas líneas)", value=default_labels, height=100)

if st.button("🔎 Clasificar", use_container_width=True):
    labels = parse_labels(raw_labels)

    if not texto.strip():
        st.warning("Por favor ingresa un texto.")
        st.stop()
    if not labels:
        st.warning("Por favor ingresa al menos una etiqueta.")
        st.stop()

    with st.spinner("Clasificando..."):
        try:
            result = clf(
                sequences=texto,
                candidate_labels=labels,
                multi_label=multi_label,
                hypothesis_template=hypothesis_template
            )
        except Exception as e:
            st.error(f"Ocurrió un error al clasificar: {e}")
            st.stop()

    # El pipeline devuelve labels y scores alineados
    out_labels = result["labels"]
    out_scores = result["scores"]

    # Orden opcional
    if sort_desc:
        # result ya viene ordenado desc si multi_label=False;
        # en multi_label=True puede variar, por eso reordenamos igual:
        pairs = sorted(zip(out_labels, out_scores), key=lambda x: x[1], reverse=True)
        out_labels, out_scores = zip(*pairs)

    df = to_df(list(out_labels), list(out_scores))

    st.subheader("Resultados")
    st.dataframe(df, use_container_width=True)

    st.subheader("Gráfico de barras")
    # st.bar_chart espera índices/columnas claras:
    chart_df = df.set_index("Etiqueta")
    st.bar_chart(chart_df)

    st.info(
        "ℹ️ **Zero-Shot (NLI)**: El modelo evalúa si el texto implica hipótesis del tipo "
        f"“{hypothesis_template.format('etiqueta')}”.\n"
        "Activa *Multi-label* si el texto puede pertenecer a varias categorías a la vez."
    )

st.markdown("---")
st.caption("Modelo: facebook/bart-large-mnli • Librería: 🤗 transformers • UI: Streamlit")
