import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import joblib
import matplotlib.pyplot as plt
from collections import Counter

# Dependencias opcionales: la app debe correr aunque RoBERTuito, Gemini o NLTK fallen.
try:
    import seaborn as sns
    SEABORN_DISPONIBLE = True
except Exception:
    sns = None
    SEABORN_DISPONIBLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    STOPWORDS_ES = set(stopwords.words("spanish"))
except Exception:
    STOPWORDS_ES = {
        "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
        "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante",
        "e", "el", "ella", "ellas", "ellos", "en", "entre", "era", "eran",
        "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estado",
        "estas", "este", "esto", "estos", "ha", "han", "hasta", "hay", "la",
        "las", "le", "les", "lo", "los", "mas", "me", "mi", "mis", "mucho",
        "muy", "no", "nos", "o", "para", "pero", "por", "porque", "que",
        "se", "sin", "sobre", "su", "sus", "tambien", "te", "tiene", "tienen",
        "todo", "un", "una", "uno", "y", "ya"
    }

try:
    from pysentimiento import create_analyzer
    PYSENTIMIENTO_INSTALADO = True
except Exception:
    create_analyzer = None
    PYSENTIMIENTO_INSTALADO = False

try:
    import google.generativeai as genai
    GEMINI_INSTALADO = True
except Exception:
    genai = None
    GEMINI_INSTALADO = False

try:
    from ftfy import fix_text
except Exception:
    def fix_text(texto):
        return texto

st.set_page_config(
    page_title="Sistema Inteligente de Desempeño Docente",
    layout="wide"
)

st.title("📊 Sistema Inteligente de Desempeño Docente")
st.caption("Modelo propio SVM + TF-IDF entrenado con etiquetas humanas, análisis Likert, benchmark RoBERTuito y recomendaciones de mejora.")

# =========================
# CARGA DE MODELOS
# =========================
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_percepcion_docente_svm_etiqueta_manual.pkl")

@st.cache_resource
def cargar_robertuito():
    if not PYSENTIMIENTO_INSTALADO or create_analyzer is None:
        return None
    return create_analyzer(task="sentiment", lang="es")

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error("No se pudo cargar el modelo propio. Verifica que el archivo 'modelo_percepcion_docente_svm_etiqueta_manual.pkl' esté en el repositorio.")
    st.exception(e)
    st.stop()

try:
    analyzer = cargar_robertuito()
except Exception:
    # RoBERTuito queda como benchmark opcional. Si no carga, la app sigue funcionando.
    analyzer = None

# Gemini desde Streamlit Secrets. No se solicita al usuario en pantalla.
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    GEMINI_API_KEY = None

# =========================
# CONFIGURACIÓN
# =========================
comentarios_invalidos = {
    "ninguno", "ninguna", "ningun comentario", "ningún comentario",
    "sin comentarios", "sin comentario", "no aplica", "n/a", "na",
    "ok", "todo bien", "sin novedad", ".", "..", "...", "-", "--",
    "no hay comentarios", "ningun", "ningún", "de acuerdo"
}

palabras_excluir = set(STOPWORDS_ES) | set([
    "docente", "docentes", "profesor", "profesora", "estudiante", "estudiantes",
    "clase", "clases", "materia", "curso", "tema", "temas", "modulo", "módulo",
    "excelente", "bueno", "buena", "buen", "muy", "bien", "gracias",
    "ninguno", "ninguna", "ok", "maestro", "maestra", "profe",
    "jaime", "johanna", "johana", "marcelo", "santiago", "pablo", "lorena", "fabián", "fabian",
    "siempre", "sido", "ser", "hacer", "forma", "parte", "puede", "podría", "considero"
])

TEMAS_PEDAGOGICOS = {
    "Claridad y explicación": ["explica", "explicación", "claro", "claridad", "entiende", "entender", "comprensión", "dudas", "confuso", "confusa"],
    "Ejercicios prácticos": ["ejercicio", "ejercicios", "práctica", "practica", "práctico", "practico", "casos", "ejemplos", "aplicación", "aplicar"],
    "Retroalimentación": ["retroalimentación", "feedback", "corrección", "corregir", "calificación", "califica", "comentarios", "devolución"],
    "Acompañamiento individual": ["individual", "personalizada", "personalizado", "tutoría", "tutoria", "acompañamiento", "asesoría", "asesoria", "uno a uno", "individuales"],
    "Metodología": ["metodología", "metodologia", "dinámica", "dinamica", "didáctica", "didactica", "participación", "participacion", "grupo", "grupal"],
    "Ritmo y tiempo": ["rápido", "rapido", "lento", "tiempo", "ritmo", "apresurado", "demora", "plazo", "entrega"],
    "Plataforma y recursos": ["plataforma", "aula", "virtual", "material", "diapositiva", "diapositivas", "recurso", "recursos", "contenido", "contenidos"],
    "Organización": ["organización", "organizacion", "orden", "desorden", "planificación", "planificacion", "estructura", "cronograma"],
    "Evaluación": ["evaluación", "evaluacion", "examen", "prueba", "tarea", "trabajo", "rúbrica", "rubrica", "nota", "calificar"]
}

# Patrones críticos: capa complementaria de seguridad interpretativa.
# No reemplaza al modelo SVM; ayuda a marcar comentarios que requieren revisión humana.
PATRONES_CRITICOS = [
    "no explica", "no se entiende", "no entiendo", "no responde", "no resuelve",
    "no domina", "no sabe", "solo lee", "lee diapositivas", "mala metodología",
    "mala metodologia", "desorganizado", "desorganizada", "muy rápido", "muy rapido",
    "va rápido", "va rapido", "califica por caras", "no retroalimenta",
    "sin retroalimentación", "sin retroalimentacion", "poca práctica", "poca practica",
    "falta práctica", "falta practica", "no es claro", "no es clara", "confuso",
    "confusa", "no aclara", "no aclara dudas", "no ayuda", "poco claro", "poco clara",
    "no contesta", "no responde dudas", "no hay retroalimentación", "no hay retroalimentacion",
    "no da ejemplos", "no hace ejercicios", "no realiza ejercicios", "se demora en responder",
    "no cumple", "no cumple horarios", "injusto", "injusta", "favoritismo"
]


def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = fix_text(str(texto))
    texto = texto.replace("Â", " ").replace("\xa0", " ").lower().strip()
    texto = re.sub(r"[^a-záéíóúñü\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def es_valido(texto):
    texto = limpiar_texto(texto)
    if texto in comentarios_invalidos:
        return False
    if len(texto) < 6:
        return False
    if len(texto.split()) < 2:
        return False
    return True


def analizar_sentimiento_robertuito(texto):
    if analyzer is None:
        return "No disponible", np.nan
    try:
        pred = analyzer.predict(texto)
        mapa = {"POS": "Positivo", "NEU": "Neutro", "NEG": "Negativo"}
        return mapa.get(pred.output, "Neutro"), float(pred.probas[pred.output])
    except Exception:
        return "No disponible", np.nan


def limpiar_nombre_columna(col):
    col = str(col).strip()
    col = re.sub(r",+$", "", col)
    col = re.sub(r"\s+", " ", col)
    return col.strip()


def limpiar_columnas_dataframe(df):
    df = df.copy()
    df.columns = [limpiar_nombre_columna(c) for c in df.columns]
    df = df.loc[:, [c for c in df.columns if str(c).strip().lower() not in ["", "nan", "none"]]]
    df = df.dropna(axis=1, how="all")
    return df


def leer_csv_robusto(archivo):
    """
    Lee CSV/XLSX de forma robusta. Para CSV prioriza separador punto y coma (;).
    """
    nombre = archivo.name.lower()

    if nombre.endswith((".xlsx", ".xls")):
        archivo.seek(0)
        return limpiar_columnas_dataframe(pd.read_excel(archivo))

    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    separadores = [";", ",", "\t", "|"]
    columnas_esperadas = {"Q Type", "Answer", "Answer Match", "# Responses", "Etiqueta Manual"}

    candidatos = []
    errores = []

    for enc in encodings:
        for sep in separadores:
            try:
                archivo.seek(0)
                df_tmp = pd.read_csv(
                    archivo,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    dtype=str,
                    on_bad_lines="skip"
                )
                df_tmp = limpiar_columnas_dataframe(df_tmp)
                coincidencias = len(columnas_esperadas.intersection(set(df_tmp.columns)))
                score = coincidencias * 10 + df_tmp.shape[1]
                candidatos.append((score, df_tmp, enc, sep))
            except Exception as e:
                errores.append((enc, sep, str(e)[:120]))

    if not candidatos:
        raise ValueError(f"No se pudo leer el archivo. Errores probados: {errores[:8]}")

    candidatos = sorted(candidatos, key=lambda x: x[0], reverse=True)
    return candidatos[0][1]


def detectar_temas(texto):
    encontrados = []
    texto = limpiar_texto(texto)
    for tema, claves in TEMAS_PEDAGOGICOS.items():
        if any(clave in texto for clave in claves):
            encontrados.append(tema)
    return encontrados


def detectar_senal_critica(texto):
    """Detecta frases críticas que pueden requerir revisión académica aunque el modelo clasifique positivo."""
    texto = limpiar_texto(texto)
    return any(patron in texto for patron in PATRONES_CRITICOS)


def decision_final(row):
    """Combina modelo SVM, señales críticas, benchmark RoBERTuito y confianza para una decisión interpretativa."""
    if row["percepcion_modelo"] == "Negativo":
        return "Negativo"

    if row.get("senal_critica", False):
        return "Revisión académica"

    if (
        row.get("percepcion_modelo") == "Positivo"
        and row.get("sentimiento_robertuito") == "Negativo"
    ):
        return "Revisión académica"

    if pd.notna(row.get("confianza_modelo_svm")) and row.get("confianza_modelo_svm") < 0.60:
        return "Revisión por baja confianza"

    return row["percepcion_modelo"]


def resumen_temas(df):
    temas = []
    for lista in df["temas_mejora"]:
        temas.extend(lista)
    return pd.DataFrame(Counter(temas).most_common(), columns=["Tema pedagógico", "Frecuencia"])


def obtener_prediccion_y_confianza(modelo, textos):
    """Devuelve predicción del modelo propio y confianza. Si el modelo fue calibrado usa predict_proba; si no, usa margen SVM normalizado."""
    pred = modelo.predict(textos)
    clases = list(getattr(modelo, "classes_", []))
    if not clases and hasattr(modelo, "named_steps") and "modelo" in modelo.named_steps:
        clases = list(getattr(modelo.named_steps["modelo"], "classes_", []))

    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(textos)
        conf = proba.max(axis=1)
        return pred, conf

    if hasattr(modelo, "decision_function"):
        scores = modelo.decision_function(textos)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            # Caso binario. Se convierte el margen a pseudo-probabilidad.
            conf = 1 / (1 + np.exp(-np.abs(scores)))
        else:
            # Softmax sobre márgenes para aproximar confianza cuando el SVM no está calibrado.
            scores = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            conf = probs.max(axis=1)
        return pred, conf

    return pred, np.repeat(np.nan, len(textos))

# =========================
# ENTRADA
# =========================
archivo = st.sidebar.file_uploader("Sube la evaluación del docente", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("---")
st.sidebar.write("**Modelo principal:** SVM Lineal + TF-IDF")
st.sidebar.write("**Entrenamiento:** etiquetas humanas")
st.sidebar.write("**Benchmark:** RoBERTuito opcional" if analyzer is not None else "**Benchmark:** No disponible")
st.sidebar.write("**Gemini:** automático con Streamlit Secrets" if GEMINI_API_KEY else "**Gemini:** no configurado en Secrets")

if archivo is None:
    st.info("Sube un archivo CSV de evaluación docente para iniciar el análisis.")
    st.stop()

df_raw = leer_csv_robusto(archivo)
df_raw.columns = df_raw.columns.str.strip()

st.subheader("1. Vista inicial del archivo")
st.write("Este bloque permite verificar que el archivo fue cargado correctamente y que contiene las columnas esperadas del reporte institucional.")
st.dataframe(df_raw.head(), use_container_width=True)

COL_TIPO = "Q Type"
COL_RESPUESTA = "Answer"
COL_RESPUESTA_LIKERT = "Answer Match"
COL_RESPUESTAS = "# Responses"

if COL_TIPO not in df_raw.columns or COL_RESPUESTA not in df_raw.columns:
    st.error("El archivo no tiene las columnas esperadas: Q Type y Answer.")
    st.stop()

# =========================
# 2. ANÁLISIS LIKERT
# =========================
st.subheader("2. Evaluación cuantitativa (Likert)")
st.write("Este bloque calcula la valoración cuantitativa del docente a partir de preguntas cerradas respondidas por estudiantes en escala Likert. Este indicador complementa el análisis cualitativo de comentarios.")

mapa_likert = {
    "Totalmente de acuerdo": 5,
    "De acuerdo": 4,
    "Neutral": 3,
    "Ni de acuerdo ni en desacuerdo": 3,
    "En desacuerdo": 2,
    "Totalmente en desacuerdo": 1,
}
orden_likert = ["Totalmente en desacuerdo", "En desacuerdo", "Neutral", "De acuerdo", "Totalmente de acuerdo"]

df_qt = df_raw[df_raw[COL_TIPO].astype(str).str.upper().str.strip() == "LIK"].copy()
nota_global = np.nan
total_respuestas = 0

if not df_qt.empty and COL_RESPUESTA_LIKERT in df_qt.columns:
    df_qt["valor_likert"] = df_qt[COL_RESPUESTA_LIKERT].astype(str).str.strip().map(mapa_likert)
    df_qt = df_qt.dropna(subset=["valor_likert"])

    if not df_qt.empty:
        if COL_RESPUESTAS in df_qt.columns:
            df_qt[COL_RESPUESTAS] = pd.to_numeric(df_qt[COL_RESPUESTAS], errors="coerce").fillna(0)
            total_respuestas = int(df_qt[COL_RESPUESTAS].sum())
            nota_global = (df_qt["valor_likert"] * df_qt[COL_RESPUESTAS]).sum() / total_respuestas if total_respuestas > 0 else df_qt["valor_likert"].mean()
        else:
            total_respuestas = len(df_qt)
            nota_global = df_qt["valor_likert"].mean()

        c1, c2 = st.columns(2)
        c1.metric("Nota global docente", f"{nota_global:.2f} / 5")
        c2.metric("Respuestas Likert válidas", f"{total_respuestas:,}")

        df_likert_resumen = (
            df_qt.groupby([COL_RESPUESTA_LIKERT, "valor_likert"], as_index=False)[COL_RESPUESTAS]
            .sum()
            .rename(columns={COL_RESPUESTA_LIKERT: "Categoría Likert", "valor_likert": "Puntaje", COL_RESPUESTAS: "Frecuencia"})
        )
        df_orden = pd.DataFrame({"Categoría Likert": orden_likert, "Puntaje": [mapa_likert[x] for x in orden_likert]})
        df_likert_resumen = df_orden.merge(df_likert_resumen, on=["Categoría Likert", "Puntaje"], how="left")
        df_likert_resumen["Frecuencia"] = df_likert_resumen["Frecuencia"].fillna(0).astype(int)

        st.write("### Distribución de respuestas Likert")
        fig_lik, ax_lik = plt.subplots(figsize=(9, 4))
        sns.barplot(data=df_likert_resumen, x="Categoría Likert", y="Frecuencia", order=orden_likert, ax=ax_lik)
        ax_lik.set_title(f"Distribución de respuestas Likert | Nota global: {nota_global:.2f}/5")
        ax_lik.set_xlabel("Categoría Likert")
        ax_lik.set_ylabel("Frecuencia")
        ax_lik.tick_params(axis="x", rotation=20)
        for container in ax_lik.containers:
            ax_lik.bar_label(container, fmt="%.0f")
        st.pyplot(fig_lik)
        st.dataframe(df_likert_resumen, use_container_width=True)

        if nota_global >= 4.5:
            interp_likert = "El docente presenta una valoración cuantitativa sobresaliente y consistente."
        elif nota_global >= 4.0:
            interp_likert = "El docente presenta una valoración favorable con oportunidades puntuales de mejora."
        elif nota_global >= 3.5:
            interp_likert = "El docente presenta una valoración intermedia; se recomienda seguimiento preventivo."
        else:
            interp_likert = "El docente presenta señales cuantitativas que requieren intervención académica."
        st.success(f"Interpretación cuantitativa: {interp_likert}")
    else:
        st.warning("Se encontraron filas Likert, pero las respuestas no coincidieron con el mapa de escala esperado.")
else:
    st.warning("No se encontraron respuestas Likert válidas en el archivo.")

# =========================
# 3. LIMPIEZA DE COMENTARIOS
# =========================
df_re = df_raw[df_raw[COL_TIPO].astype(str).str.upper().str.strip() == "RE"].copy()
df_re["comentario_limpio"] = df_re[COL_RESPUESTA].apply(limpiar_texto)
df_re["comentario_valido"] = df_re["comentario_limpio"].apply(es_valido)

comentarios_antes = len(df_re)
df_re = df_re[df_re["comentario_valido"]].copy()
comentarios_validos = len(df_re)
comentarios_eliminados = comentarios_antes - comentarios_validos

st.subheader("3. Limpieza de comentarios")
c1, c2, c3 = st.columns(3)
c1.metric("Comentarios abiertos", comentarios_antes)
c2.metric("Comentarios válidos", comentarios_validos)
c3.metric("Eliminados", comentarios_eliminados)

resumen_limpieza = pd.DataFrame({
    "Estado": ["Comentarios originales", "Comentarios válidos", "Comentarios eliminados"],
    "Cantidad": [comentarios_antes, comentarios_validos, comentarios_eliminados]
})
fig_limp, ax_limp = plt.subplots(figsize=(7, 4))
sns.barplot(data=resumen_limpieza, x="Estado", y="Cantidad", ax=ax_limp)
ax_limp.set_title("Proceso de depuración de comentarios")
ax_limp.set_xlabel("Estado")
ax_limp.set_ylabel("Cantidad")
ax_limp.tick_params(axis="x", rotation=15)
for container in ax_limp.containers:
    ax_limp.bar_label(container, fmt="%.0f")
st.pyplot(fig_limp)

st.info("Se eliminaron comentarios que no aportan valor analítico, como 'ninguno', 'N/A', 'ok', puntos o respuestas demasiado cortas. Esto evita que el modelo confunda ausencia de opinión con una percepción real del estudiante.")

if comentarios_validos == 0:
    st.warning("No hay comentarios válidos para analizar.")
    st.stop()

# =========================
# 4. MODELO PROPIO SVM
# =========================
st.subheader("4. Análisis de percepción con modelo propio")
st.write("El modelo principal del sistema es un clasificador SVM Lineal con representación TF-IDF, entrenado con etiquetas humanas. Este modelo es el encargado de evaluar la percepción estudiantil en los comentarios.")

pred_svm, conf_svm = obtener_prediccion_y_confianza(modelo, df_re["comentario_limpio"])
df_re["percepcion_modelo"] = pred_svm
df_re["confianza_modelo_svm"] = conf_svm
df_re["senal_critica"] = df_re["comentario_limpio"].apply(detectar_senal_critica)

conteo_modelo = df_re["percepcion_modelo"].value_counts()
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=df_re, x="percepcion_modelo", order=["Positivo", "Neutro", "Negativo"], ax=ax)
ax.set_title("Distribución de percepción estudiantil - Modelo propio SVM")
ax.set_xlabel("Clasificación del modelo propio")
ax.set_ylabel("Cantidad de comentarios")
st.pyplot(fig)

predominante = conteo_modelo.idxmax()
porcentaje_predominante = conteo_modelo.max() / conteo_modelo.sum() * 100
conf_media_svm = np.nanmean(df_re["confianza_modelo_svm"]) * 100
st.info(f"La percepción predominante según el modelo propio es **{predominante}** con **{porcentaje_predominante:.1f}%** de los comentarios válidos. La confianza promedio del modelo propio es **{conf_media_svm:.1f}%**.")

# =========================
# 5. ROBERTUITO BENCHMARK
# =========================
st.subheader("5. Comparación con RoBERTuito")
st.write("RoBERTuito se utiliza como benchmark externo de análisis de sentimiento en español. No es el modelo principal del sistema; sirve para comparar la salida del modelo propio frente a un modelo preentrenado.")

resultados_robertuito = df_re["comentario_limpio"].apply(analizar_sentimiento_robertuito)
df_re["sentimiento_robertuito"] = [r[0] for r in resultados_robertuito]
df_re["confianza_robertuito"] = [r[1] for r in resultados_robertuito]
df_re["decision_final"] = df_re.apply(decision_final, axis=1)

fig2, ax2 = plt.subplots(figsize=(7, 4))
sns.countplot(data=df_re, x="sentimiento_robertuito", order=["Positivo", "Neutro", "Negativo"], ax=ax2)
ax2.set_title("Distribución de sentimiento con RoBERTuito")
ax2.set_xlabel("Sentimiento RoBERTuito")
ax2.set_ylabel("Cantidad")
st.pyplot(fig2)

mask_robertuito = df_re["sentimiento_robertuito"].isin(["Positivo", "Neutro", "Negativo"])
if mask_robertuito.any():
    coinciden = (df_re.loc[mask_robertuito, "percepcion_modelo"] == df_re.loc[mask_robertuito, "sentimiento_robertuito"]).sum()
    concordancia = coinciden / mask_robertuito.sum() * 100
    conf_media_robertuito = df_re.loc[mask_robertuito, "confianza_robertuito"].mean() * 100
else:
    concordancia = np.nan
    conf_media_robertuito = np.nan

m1, m2, m3 = st.columns(3)
m1.metric("Concordancia SVM vs RoBERTuito", "No disponible" if pd.isna(concordancia) else f"{concordancia:.1f}%")
m2.metric("Confianza promedio SVM", f"{conf_media_svm:.1f}%")
m3.metric("Confianza promedio RoBERTuito", "No disponible" if pd.isna(conf_media_robertuito) else f"{conf_media_robertuito:.1f}%")
st.info("La confianza SVM corresponde al modelo propio desarrollado con etiquetas humanas. La confianza RoBERTuito corresponde al modelo externo usado como benchmark. En caso de discrepancias, se recomienda revisión humana del comentario.")

st.info("La **decisión final del sistema** combina la clasificación SVM, la confianza del modelo, la comparación con RoBERTuito y una capa de señales críticas pedagógicas. Esta decisión final no reemplaza la revisión humana; ayuda a priorizar comentarios que requieren análisis académico.")

# =========================
# 6. TEMAS PEDAGÓGICOS
# =========================
st.subheader("6. Temas pedagógicos y oportunidades de mejora")
st.write("Esta capa detecta temas de mejora incluso cuando el comentario general es positivo o neutro. Por ejemplo, un estudiante puede valorar bien al docente, pero pedir más ejercicios prácticos o más acompañamiento individual.")

df_re["temas_mejora"] = df_re["comentario_limpio"].apply(detectar_temas)
df_temas = resumen_temas(df_re)

if not df_temas.empty:
    fig_temas, ax_temas = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_temas.head(10), x="Frecuencia", y="Tema pedagógico", ax=ax_temas)
    ax_temas.set_title("Temas pedagógicos detectados en comentarios")
    st.pyplot(fig_temas)
    st.dataframe(df_temas, use_container_width=True)
else:
    st.warning("No se detectaron temas pedagógicos con las reglas actuales.")

# =========================
# 7. PALABRAS FRECUENTES
# =========================
st.subheader("7. Palabras relevantes más frecuentes")
texto_total = " ".join(df_re["comentario_limpio"].tolist())
palabras = texto_total.split()
palabras_filtradas = [p for p in palabras if p not in palabras_excluir and len(p) > 3]
df_palabras = pd.DataFrame(Counter(palabras_filtradas).most_common(20), columns=["Palabra", "Frecuencia"])

if not df_palabras.empty:
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df_palabras, x="Frecuencia", y="Palabra", ax=ax3)
    ax3.set_title("Top palabras relevantes")
    st.pyplot(fig3)
    top_terms = ", ".join(df_palabras.head(5)["Palabra"].tolist())
    st.info(f"Las palabras más repetidas después de eliminar términos genéricos son: **{top_terms}**.")
else:
    st.warning("No se encontraron palabras relevantes suficientes después de la limpieza.")

# =========================
# 8. COMENTARIOS ALERTA
# =========================
st.subheader("8. Comentarios que requieren revisión")
df_alerta = df_re[
    (df_re["decision_final"].isin(["Negativo", "Revisión académica", "Revisión por baja confianza"]))
    | (df_re["sentimiento_robertuito"] == "Negativo")
    | (df_re["temas_mejora"].apply(len) > 0)
].copy()

if not df_alerta.empty:
    df_mostrar = df_alerta[[
        COL_RESPUESTA,
        "percepcion_modelo",
        "decision_final",
        "senal_critica",
        "confianza_modelo_svm",
        "sentimiento_robertuito",
        "confianza_robertuito",
        "temas_mejora"
    ]].head(15).copy()
    df_mostrar["confianza_modelo_svm"] = (df_mostrar["confianza_modelo_svm"] * 100).round(1).astype(str) + "%"
    df_mostrar["confianza_robertuito"] = (df_mostrar["confianza_robertuito"] * 100).round(1).astype(str) + "%"
    df_mostrar = df_mostrar.rename(columns={
        COL_RESPUESTA: "Comentario original",
        "percepcion_modelo": "Clasificación SVM",
        "decision_final": "Decisión final del sistema",
        "senal_critica": "Señal crítica detectada",
        "confianza_modelo_svm": "Confianza SVM",
        "sentimiento_robertuito": "Benchmark RoBERTuito",
        "confianza_robertuito": "Confianza RoBERTuito",
        "temas_mejora": "Temas pedagógicos detectados"
    })
    st.dataframe(df_mostrar, use_container_width=True)
    st.info("Estos comentarios requieren revisión porque contienen señales de inconformidad, baja confianza, discrepancias entre modelos o temas pedagógicos accionables. La clasificación SVM es la salida del modelo principal; la decisión final agrega una capa de seguridad interpretativa. No constituye una decisión automática.")
else:
    st.success("No se detectaron comentarios prioritarios según el modelo y la capa pedagógica.")

# =========================
# 9. DIAGNÓSTICO
# =========================
st.subheader("9. Diagnóstico interpretativo")
porc_neg = (df_re["percepcion_modelo"].eq("Negativo").mean() * 100)
porc_neu = (df_re["percepcion_modelo"].eq("Neutro").mean() * 100)
porc_pos = (df_re["percepcion_modelo"].eq("Positivo").mean() * 100)
porc_revision = (df_re["decision_final"].isin(["Revisión académica", "Revisión por baja confianza"]).mean() * 100)
porc_neg_final = (df_re["decision_final"].eq("Negativo").mean() * 100)

if porc_neg_final >= 30 or porc_revision >= 30:
    nivel_alerta = "Alto"
    interpretacion = "Se recomienda revisión académica prioritaria y acompañamiento docente."
elif porc_neg_final >= 15 or porc_neu >= 40 or porc_revision >= 15:
    nivel_alerta = "Medio"
    interpretacion = "Se recomienda acompañamiento pedagógico y seguimiento en la siguiente evaluación."
else:
    nivel_alerta = "Bajo"
    interpretacion = "Se recomienda seguimiento regular y mantenimiento de buenas prácticas."

nota_texto = "No disponible" if pd.isna(nota_global) else f"{nota_global:.2f}/5"
temas_principales = df_temas.head(5)["Tema pedagógico"].tolist() if not df_temas.empty else []

st.markdown(f"""
**Resumen del análisis**

- Nota cuantitativa Likert: **{nota_texto}**
- Percepción positiva según modelo propio: **{porc_pos:.1f}%**
- Percepción neutra según modelo propio: **{porc_neu:.1f}%**
- Percepción negativa según modelo propio: **{porc_neg:.1f}%**
- Comentarios marcados para revisión por decisión final: **{porc_revision:.1f}%**
- Confianza promedio del modelo propio SVM: **{conf_media_svm:.1f}%**
- Concordancia con RoBERTuito: **{concordancia:.1f}%**
- Temas pedagógicos principales: **{', '.join(temas_principales) if temas_principales else 'No disponible'}**
- Nivel de alerta académica: **{nivel_alerta}**

**Interpretación:**  
{interpretacion}
""")

# =========================
# 10. GEMINI
# =========================
st.subheader("10. Recomendaciones con Gemini")

if GEMINI_API_KEY and GEMINI_INSTALADO:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        modelos_disponibles = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-flash-latest", "gemini-pro-latest"]
        prompt = f"""
Actúa como experto en evaluación docente, analítica educativa y mejora académica.

Resultados del análisis:
- Modelo principal: SVM Lineal + TF-IDF entrenado con etiquetas humanas.
- Nota cuantitativa Likert: {nota_texto}
- Comentarios válidos: {comentarios_validos}
- Comentarios eliminados: {comentarios_eliminados}
- Percepción positiva modelo propio: {porc_pos:.1f}%
- Percepción neutra modelo propio: {porc_neu:.1f}%
- Percepción negativa modelo propio: {porc_neg:.1f}%
- Comentarios marcados para revisión por decisión final: {porc_revision:.1f}%
- Confianza promedio SVM: {conf_media_svm:.1f}%
- Confianza promedio RoBERTuito: {conf_media_robertuito:.1f}%
- Concordancia con RoBERTuito: {concordancia:.1f}%
- Nivel de alerta: {nivel_alerta}
- Temas pedagógicos detectados: {df_temas.head(10).to_dict(orient='records') if not df_temas.empty else []}
- Palabras frecuentes: {df_palabras.head(10).to_dict(orient='records') if not df_palabras.empty else []}
- Comentarios prioritarios: {df_alerta[COL_RESPUESTA].head(5).tolist() if not df_alerta.empty else []}
- Señales críticas detectadas: {int(df_re['senal_critica'].sum())}

Genera:
1. Resumen ejecutivo.
2. Qué están reflejando los estudiantes.
3. Fortalezas del docente.
4. Aspectos específicos que debe mejorar, incluyendo temas pedagógicos aunque aparezcan en comentarios positivos o neutros.
5. 10 recomendaciones accionables.
6. Cierre ético indicando que esto apoya decisiones, pero no reemplaza revisión humana.
Usa lenguaje profesional, orientado a mejora continua y evita tono punitivo.
"""
        respuesta_texto = None
        modelo_activo = None
        ultimo_error = None
        for nombre_modelo in modelos_disponibles:
            try:
                model_ia = genai.GenerativeModel(nombre_modelo)
                response = model_ia.generate_content(prompt)
                respuesta_texto = response.text
                modelo_activo = nombre_modelo
                break
            except Exception as e:
                ultimo_error = e
                continue
        if respuesta_texto:
            st.success(f"Recomendaciones generadas con: {modelo_activo}")
            st.write(respuesta_texto)
        else:
            st.error(f"No se pudo generar recomendaciones. Último error: {ultimo_error}")
    except Exception as e:
        st.error(f"Error general con Gemini: {e}")
else:
    st.info("Gemini no está configurado en Streamlit Secrets. La app funciona sin recomendaciones generativas automáticas.")

# =========================
# 11. DESCARGA
# =========================
st.subheader("11. Descargar resultados")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Descargar resultados CSV",
        data=df_re.to_csv(index=False, encoding="utf-8-sig"),
        file_name="resultados_streamlit_docente.csv",
        mime="text/csv",
    )
with col2:
    st.download_button(
        "Descargar temas pedagógicos CSV",
        data=df_temas.to_csv(index=False, encoding="utf-8-sig") if not df_temas.empty else "",
        file_name="temas_pedagogicos_detectados.csv",
        mime="text/csv",
    )
