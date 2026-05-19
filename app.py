import os
import re
import io
import urllib.parse
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Dependencias opcionales: la app no debe caerse si no están disponibles.
try:
    from ftfy import fix_text
except Exception:
    def fix_text(texto):
        return texto

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

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Sistema Inteligente de Desempeño Docente",
    layout="wide"
)

st.title("📊 Sistema Inteligente de Desempeño Docente")
st.caption(
    "Herramienta de soporte académico: análisis Likert, NLP con TF-IDF, modelos supervisados, "
    "benchmark con RoBERTuito, umbral académico y revisión experta."
)

COMENTARIOS_INVALIDOS = {
    "ninguno", "ninguna", "ningun comentario", "ningún comentario",
    "sin comentarios", "sin comentario", "no aplica", "n/a", "na",
    "ok", "todo bien", "sin novedad", ".", "..", "...", "-", "--",
    "no hay comentarios", "ningun", "ningún", "de acuerdo"
}

PALABRAS_EXCLUIR = set(STOPWORDS_ES) | {
    "docente", "docentes", "profesor", "profesora", "estudiante", "estudiantes",
    "clase", "clases", "materia", "curso", "tema", "temas", "modulo", "módulo",
    "excelente", "bueno", "buena", "buen", "muy", "bien", "gracias",
    "ninguno", "ninguna", "ok", "maestro", "maestra", "profe",
    "siempre", "sido", "ser", "hacer", "forma", "parte", "puede", "podría", "considero"
}

TEMAS_PEDAGOGICOS = {
    "Claridad y explicación": ["explica", "explicación", "claro", "claridad", "entiende", "entender", "comprensión", "dudas", "confuso", "confusa", "poco claro", "poco clara"],
    "Ejercicios prácticos": ["ejercicio", "ejercicios", "práctica", "practica", "práctico", "practico", "casos", "ejemplos", "aplicación", "aplicar"],
    "Retroalimentación": ["retroalimentación", "retroalimentacion", "feedback", "corrección", "corregir", "calificación", "califica", "comentarios", "devolución"],
    "Acompañamiento individual": ["individual", "personalizada", "personalizado", "tutoría", "tutoria", "acompañamiento", "asesoría", "asesoria", "uno a uno", "individuales"],
    "Metodología": ["metodología", "metodologia", "dinámica", "dinamica", "didáctica", "didactica", "participación", "participacion", "grupo", "grupal"],
    "Ritmo y tiempo": ["rápido", "rapido", "lento", "tiempo", "ritmo", "apresurado", "demora", "plazo", "entrega"],
    "Plataforma y recursos": ["plataforma", "aula", "virtual", "material", "diapositiva", "diapositivas", "recurso", "recursos", "contenido", "contenidos"],
    "Organización": ["organización", "organizacion", "orden", "desorden", "planificación", "planificacion", "estructura", "cronograma"],
    "Evaluación": ["evaluación", "evaluacion", "examen", "prueba", "tarea", "trabajo", "rúbrica", "rubrica", "nota", "calificar"],
    "Comunicación": ["responde", "responder", "mensaje", "correo", "comunicación", "comunicacion", "foro", "consulta", "consultas"],
    "Nivel académico": ["profundidad", "básico", "basico", "rigor", "maestría", "maestria", "posgrado", "superficial"]
}

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

CONECTORES_CONTRASTE = ["pero", "sin embargo", "aunque", "no obstante", "a pesar de", "solo que"]

MAPA_LIKERT = {
    "Totalmente de acuerdo": 5,
    "De acuerdo": 4,
    "Neutral": 3,
    "Ni de acuerdo ni en desacuerdo": 3,
    "En desacuerdo": 2,
    "Totalmente en desacuerdo": 1,
}
ORDEN_LIKERT = ["Totalmente en desacuerdo", "En desacuerdo", "Neutral", "De acuerdo", "Totalmente de acuerdo"]

# ============================================================
# FUNCIONES DE CARGA
# ============================================================
@st.cache_resource
def cargar_modelos_propios():
    """
    Carga modelos propios entrenados en el notebook.
    Orden de prioridad:
    1. Un archivo tipo bundle/diccionario con varios modelos.
    2. Archivos individuales de regresión logística, random forest y SVM.
    3. Archivo histórico SVM.
    """
    candidatos_bundle = [
        "modelos_propios_docente.pkl",
        "modelos_entrenados_docente.pkl",
        "modelos_entrenados.pkl",
        "bundle_modelos_docente.pkl"
    ]

    modelos = {}
    metricas = None
    nombre_ganador = None

    for ruta in candidatos_bundle:
        if os.path.exists(ruta):
            obj = joblib.load(ruta)
            if isinstance(obj, dict):
                if "modelos" in obj and isinstance(obj["modelos"], dict):
                    modelos = obj["modelos"]
                else:
                    posibles = {k: v for k, v in obj.items() if hasattr(v, "predict")}
                    modelos = posibles
                metricas = obj.get("metricas") if "metricas" in obj else None
                nombre_ganador = obj.get("modelo_ganador") if "modelo_ganador" in obj else None
                if modelos:
                    return modelos, metricas, nombre_ganador

    archivos_individuales = {
        "Regresión Logística": "modelo_regresion_logistica.pkl",
        "Random Forest": "modelo_random_forest.pkl",
        "SVM lineal calibrado": "modelo_svm_lineal_calibrado.pkl",
        "SVM lineal calibrado": "modelo_percepcion_docente_svm_etiqueta_manual.pkl",
    }

    for nombre, ruta in archivos_individuales.items():
        if os.path.exists(ruta):
            modelos[nombre] = joblib.load(ruta)

    if not modelos and os.path.exists("modelo_percepcion_docente_svm_etiqueta_manual.pkl"):
        modelos["SVM lineal calibrado"] = joblib.load("modelo_percepcion_docente_svm_etiqueta_manual.pkl")

    return modelos, metricas, nombre_ganador


@st.cache_resource
def cargar_robertuito():
    if not PYSENTIMIENTO_INSTALADO:
        return None
    return create_analyzer(task="sentiment", lang="es")


@st.cache_data
def cargar_metricas_desde_archivo():
    posibles = [
        "metricas_modelos.csv",
        "resultados_modelos.csv",
        "comparacion_modelos.csv",
        "metricas_modelos.xlsx"
    ]
    for ruta in posibles:
        if os.path.exists(ruta):
            try:
                if ruta.endswith(".xlsx"):
                    return pd.read_excel(ruta)
                return pd.read_csv(ruta)
            except Exception:
                continue
    return None


# ============================================================
# FUNCIONES UTILITARIAS
# ============================================================
def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = fix_text(str(texto))
    texto = texto.replace("Â", " ").replace("\xa0", " ").lower().strip()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"\S+@\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúñü\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def es_valido(texto):
    texto = limpiar_texto(texto)
    if texto in COMENTARIOS_INVALIDOS:
        return False
    if len(texto) < 6:
        return False
    if len(texto.split()) < 2:
        return False
    return True


def leer_archivo_robusto(archivo):
    nombre = archivo.name.lower()
    if nombre.endswith((".xlsx", ".xls")):
        return pd.read_excel(archivo)
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            archivo.seek(0)
            return pd.read_csv(archivo, encoding=enc, sep=None, engine="python")
        except Exception:
            continue
    archivo.seek(0)
    return pd.read_csv(archivo)


def detectar_temas(texto):
    texto = limpiar_texto(texto)
    encontrados = []
    for tema, claves in TEMAS_PEDAGOGICOS.items():
        if any(clave in texto for clave in claves):
            encontrados.append(tema)
    return encontrados


def detectar_senal_critica(texto):
    texto = limpiar_texto(texto)
    return any(patron in texto for patron in PATRONES_CRITICOS)


def detectar_contraste(texto):
    texto = limpiar_texto(texto)
    return any(re.search(rf"\b{re.escape(conector)}\b", texto) for conector in CONECTORES_CONTRASTE)


def clasificar_intensidad_revision(row):
    if row.get("decision_final_sistema") == "Negativo confirmado":
        return "Alta"
    if row.get("decision_final_sistema") in ["Revisión académica", "Revisión por baja confianza"]:
        return "Media"
    if row.get("temas_pedagogicos"):
        return "Baja"
    return "Sin alerta"


def obtener_prediccion_y_confianza(modelo, textos):
    textos = list(textos)
    pred = modelo.predict(textos)

    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(textos)
        conf = proba.max(axis=1)
        return np.asarray(pred), np.asarray(conf)

    if hasattr(modelo, "decision_function"):
        scores = modelo.decision_function(textos)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            conf = 1 / (1 + np.exp(-np.abs(scores)))
        else:
            scores = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            conf = probs.max(axis=1)
        return np.asarray(pred), np.asarray(conf)

    return np.asarray(pred), np.repeat(np.nan, len(textos))


def analizar_sentimiento_robertuito(analyzer, texto):
    if analyzer is None:
        return "No disponible", np.nan
    try:
        pred = analyzer.predict(texto)
        mapa = {"POS": "Positivo", "NEU": "Neutro", "NEG": "Negativo"}
        return mapa.get(pred.output, "Neutro"), float(pred.probas[pred.output])
    except Exception:
        return "No disponible", np.nan


def resumen_temas(df):
    temas = []
    for lista in df["temas_pedagogicos"]:
        temas.extend(lista)
    return pd.DataFrame(Counter(temas).most_common(), columns=["Tema pedagógico", "Frecuencia"])


def calcular_distribucion(df, columna, prefijo):
    total = len(df)
    if total == 0 or columna not in df.columns:
        return {f"% {prefijo} Positivo": 0, f"% {prefijo} Neutro": 0, f"% {prefijo} Negativo": 0}
    return {
        f"% {prefijo} Positivo": df[columna].eq("Positivo").mean() * 100,
        f"% {prefijo} Neutro": df[columna].eq("Neutro").mean() * 100,
        f"% {prefijo} Negativo": df[columna].eq("Negativo").mean() * 100,
    }


def elegir_modelo_ganador(modelos, metricas):
    if metricas is not None and isinstance(metricas, pd.DataFrame) and not metricas.empty:
        dfm = metricas.copy()
        dfm.columns = [str(c).strip() for c in dfm.columns]
        posibles_modelo = [c for c in dfm.columns if c.lower() in ["modelo", "model", "nombre_modelo"]]
        posibles_score = [
            c for c in dfm.columns
            if c.lower() in ["f1_macro_validacion", "macro f1 validación", "macro f1 validacion", "f1_macro", "f1-score", "accuracy_validacion", "accuracy"]
        ]
        if posibles_modelo and posibles_score:
            col_modelo = posibles_modelo[0]
            col_score = posibles_score[0]
            try:
                ganador = dfm.sort_values(col_score, ascending=False).iloc[0][col_modelo]
                if ganador in modelos:
                    return ganador
            except Exception:
                pass
    if "SVM lineal calibrado" in modelos:
        return "SVM lineal calibrado"
    return list(modelos.keys())[0]


def graficar_barras(df, x, y, titulo, horizontal=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if df.empty:
        return fig
    if horizontal:
        ax.barh(df[y], df[x])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.invert_yaxis()
    else:
        ax.bar(df[x], df[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.tick_params(axis="x", rotation=20)
    ax.set_title(titulo)
    for container in ax.containers:
        try:
            ax.bar_label(container, fmt="%.0f")
        except Exception:
            pass
    fig.tight_layout()
    return fig


def construir_decision_final(row, umbral):
    pred = row.get("clasificacion_modelo_principal")
    conf = row.get("confianza_modelo_principal")
    rob = row.get("clasificacion_robertuito")

    if pd.notna(conf) and conf < umbral:
        return "Revisión por baja confianza"
    if row.get("senal_critica_detectada", False):
        return "Revisión académica"
    if rob == "Negativo" and pred != "Negativo":
        return "Revisión académica"
    if pred == "Negativo" and rob in ["Negativo", "No disponible"]:
        return "Negativo confirmado"
    return pred


def texto_bool(valor):
    return "Sí" if bool(valor) else "No"


def generar_reporte_director(nota_texto, resumen, df_temas, df_alerta, interpretacion, ruta_1, ruta_2):
    temas = df_temas.head(8).to_markdown(index=False) if not df_temas.empty else "No se detectaron temas pedagógicos recurrentes."
    comentarios = df_alerta[["Comentario original", "decision_final_sistema", "temas_pedagogicos"]].head(10).to_markdown(index=False) if not df_alerta.empty else "No se identificaron comentarios prioritarios."
    return f"""
# Reporte para Director / Jefe de Área

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Propósito:** apoyar la revisión académica del desempeño docente. La herramienta no reemplaza la decisión del experto.

## 1. Resumen ejecutivo

- Nota Likert: **{nota_texto}**
- Comentarios válidos: **{resumen['comentarios_validos']}**
- Modelo principal: **{resumen['modelo_principal']}**
- Umbral académico de confianza: **{resumen['umbral']:.2f}**
- Confianza promedio modelo principal: **{resumen['confianza_modelo']:.1f}%**
- Concordancia con RoBERTuito: **{resumen['concordancia']:.1f}%**
- Comentarios que requieren revisión: **{resumen['porc_revision']:.1f}%**
- Nivel de alerta: **{resumen['nivel_alerta']}**

## 2. Interpretación académica

{interpretacion}

## 3. Temas pedagógicos más frecuentes

{temas}

## 4. Comentarios prioritarios para revisión

{comentarios}

## 5. Dos rutas de acción sugeridas

### Ruta 1: Seguimiento preventivo
{ruta_1}

### Ruta 2: Plan de mejora académico
{ruta_2}

## 6. Cierre ético

El resultado debe ser revisado por un responsable académico. El sistema prioriza comentarios, detecta señales críticas y resume tendencias, pero no debe usarse como única fuente para decisiones institucionales.
""".strip()


def generar_reporte_docente(nota_texto, resumen, df_temas, recomendaciones_docente):
    temas = df_temas.head(8).to_markdown(index=False) if not df_temas.empty else "No se detectaron temas pedagógicos recurrentes."
    return f"""
# Reporte para Docente

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Propósito:** entregar retroalimentación formativa, no punitiva, orientada a mejora continua.

## 1. Resumen general

- Nota Likert: **{nota_texto}**
- Comentarios válidos analizados: **{resumen['comentarios_validos']}**
- Percepción positiva: **{resumen['porc_pos']:.1f}%**
- Percepción neutra: **{resumen['porc_neu']:.1f}%**
- Percepción negativa: **{resumen['porc_neg']:.1f}%**

## 2. Temas pedagógicos para fortalecer

{temas}

## 3. Recomendaciones de mejora

{recomendaciones_docente}

## 4. Nota metodológica

El análisis se basa en comentarios estudiantiles procesados mediante NLP. Los resultados deben entenderse como apoyo para reflexionar sobre la práctica docente y no como una evaluación automática definitiva.
""".strip()


def generar_recomendaciones_basicas(df_temas, nivel_alerta):
    temas = df_temas.head(5)["Tema pedagógico"].tolist() if not df_temas.empty else []
    if not temas:
        return "Mantener las buenas prácticas actuales, revisar periódicamente comentarios abiertos y reforzar comunicación académica al cierre de cada módulo."
    items = []
    for tema in temas:
        if "Claridad" in tema:
            items.append("Reforzar explicaciones con ejemplos concretos y síntesis final por clase.")
        elif "Retroalimentación" in tema:
            items.append("Aumentar la retroalimentación específica en trabajos, conectando nota, rúbrica y acciones de mejora.")
        elif "Ejercicios" in tema:
            items.append("Incorporar más casos prácticos vinculados al contexto profesional de maestría.")
        elif "Organización" in tema:
            items.append("Ordenar materiales, fechas y actividades por semana o unidad en la plataforma.")
        elif "Evaluación" in tema:
            items.append("Explicar criterios de evaluación antes de cada entrega y usar ejemplos de trabajos esperados.")
        elif "Comunicación" in tema:
            items.append("Definir tiempos de respuesta y un canal visible para dudas frecuentes.")
        else:
            items.append(f"Revisar el tema '{tema}' y acordar una acción pedagógica verificable para el siguiente módulo.")
    if nivel_alerta == "Alto":
        items.append("Agendar una revisión académica con el director o jefe de área para definir seguimiento formal.")
    return "\n".join([f"- {i}" for i in items])


# ============================================================
# SIDEBAR: ENTRADA Y CONFIGURACIÓN
# ============================================================
modelos, metricas_bundle, ganador_bundle = cargar_modelos_propios()
metricas_archivo = cargar_metricas_desde_archivo()
metricas_modelos = metricas_bundle if isinstance(metricas_bundle, pd.DataFrame) else metricas_archivo

if not modelos:
    st.error(
        "No se encontró ningún modelo propio. Incluye al menos uno de estos archivos: "
        "modelo_regresion_logistica.pkl, modelo_random_forest.pkl, modelo_svm_lineal_calibrado.pkl "
        "o modelo_percepcion_docente_svm_etiqueta_manual.pkl."
    )
    st.stop()

modelo_principal_nombre = ganador_bundle if ganador_bundle in modelos else elegir_modelo_ganador(modelos, metricas_modelos)
modelo_principal = modelos[modelo_principal_nombre]
analyzer = cargar_robertuito()

GEMINI_API_KEY = None
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    GEMINI_API_KEY = None

st.sidebar.header("Configuración")
archivo = st.sidebar.file_uploader("Sube la evaluación del docente", type=["csv", "xlsx", "xls"])
umbral_confianza = st.sidebar.slider(
    "Umbral académico de confianza",
    min_value=0.50,
    max_value=0.90,
    value=0.60,
    step=0.05,
    help="Si la confianza del modelo principal queda por debajo de este valor, el comentario pasa a revisión experta."
)
limite_comentarios_robertuito = st.sidebar.number_input(
    "Máximo de comentarios para RoBERTuito",
    min_value=10,
    max_value=500,
    value=150,
    step=10,
    help="RoBERTuito puede demorar. Limitarlo evita que Streamlit se congele con archivos grandes."
)

st.sidebar.markdown("---")
st.sidebar.write("**Modelos propios cargados:**")
for nombre in modelos.keys():
    st.sidebar.write(f"- {nombre}")
st.sidebar.write(f"**Modelo principal/global:** {modelo_principal_nombre}")
st.sidebar.write("**Representación NLP:** TF-IDF con n-grams 1, 2 y 3 si así fue entrenado en el notebook.")
st.sidebar.write("**RoBERTuito:** disponible" if analyzer is not None else "**RoBERTuito:** no disponible")
st.sidebar.write("**Gemini:** configurado" if GEMINI_API_KEY and GEMINI_INSTALADO else "**Gemini:** no configurado")

if archivo is None:
    st.info("Sube un archivo de evaluación docente para iniciar el análisis.")
    st.stop()

# ============================================================
# CARGA Y VALIDACIÓN DEL ARCHIVO
# ============================================================
df_raw = leer_archivo_robusto(archivo)
df_raw.columns = df_raw.columns.astype(str).str.strip()

COL_TIPO = "Q Type"
COL_RESPUESTA = "Answer"
COL_RESPUESTA_LIKERT = "Answer Match"
COL_RESPUESTAS = "# Responses"

if COL_TIPO not in df_raw.columns or COL_RESPUESTA not in df_raw.columns:
    st.error("El archivo debe contener las columnas institucionales 'Q Type' y 'Answer'.")
    st.dataframe(df_raw.head(), use_container_width=True)
    st.stop()

# ============================================================
# PROCESAMIENTO: LIKERT
# ============================================================
df_qt = df_raw[df_raw[COL_TIPO].astype(str).str.upper().str.strip() == "LIK"].copy()
nota_global = np.nan
total_respuestas_likert = 0
df_likert_resumen = pd.DataFrame(columns=["Categoría Likert", "Puntaje", "Frecuencia"])

if not df_qt.empty and COL_RESPUESTA_LIKERT in df_qt.columns:
    df_qt["valor_likert"] = df_qt[COL_RESPUESTA_LIKERT].astype(str).str.strip().map(MAPA_LIKERT)
    df_qt = df_qt.dropna(subset=["valor_likert"])
    if not df_qt.empty:
        if COL_RESPUESTAS in df_qt.columns:
            df_qt[COL_RESPUESTAS] = pd.to_numeric(df_qt[COL_RESPUESTAS], errors="coerce").fillna(0)
            total_respuestas_likert = int(df_qt[COL_RESPUESTAS].sum())
            if total_respuestas_likert > 0:
                nota_global = (df_qt["valor_likert"] * df_qt[COL_RESPUESTAS]).sum() / total_respuestas_likert
            else:
                nota_global = df_qt["valor_likert"].mean()
        else:
            df_qt[COL_RESPUESTAS] = 1
            total_respuestas_likert = len(df_qt)
            nota_global = df_qt["valor_likert"].mean()

        df_temp = (
            df_qt.groupby([COL_RESPUESTA_LIKERT, "valor_likert"], as_index=False)[COL_RESPUESTAS]
            .sum()
            .rename(columns={COL_RESPUESTA_LIKERT: "Categoría Likert", "valor_likert": "Puntaje", COL_RESPUESTAS: "Frecuencia"})
        )
        df_orden = pd.DataFrame({"Categoría Likert": ORDEN_LIKERT, "Puntaje": [MAPA_LIKERT[x] for x in ORDEN_LIKERT]})
        df_likert_resumen = df_orden.merge(df_temp, on=["Categoría Likert", "Puntaje"], how="left")
        df_likert_resumen["Frecuencia"] = df_likert_resumen["Frecuencia"].fillna(0).astype(int)

# ============================================================
# PROCESAMIENTO: COMENTARIOS
# ============================================================
df_re = df_raw[df_raw[COL_TIPO].astype(str).str.upper().str.strip() == "RE"].copy()
df_re["Comentario original"] = df_re[COL_RESPUESTA].astype(str)
df_re["comentario_limpio"] = df_re["Comentario original"].apply(limpiar_texto)
df_re["comentario_valido"] = df_re["comentario_limpio"].apply(es_valido)
comentarios_abiertos = len(df_re)
df_re = df_re[df_re["comentario_valido"]].copy()
comentarios_validos = len(df_re)
comentarios_eliminados = comentarios_abiertos - comentarios_validos

if comentarios_validos == 0:
    st.warning("No hay comentarios válidos para analizar.")
    st.stop()

# ============================================================
# PREDICCIONES MODELOS PROPIOS
# ============================================================
for nombre_modelo, modelo in modelos.items():
    slug = nombre_modelo.lower().replace(" ", "_").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    pred, conf = obtener_prediccion_y_confianza(modelo, df_re["comentario_limpio"])
    df_re[f"clasificacion_{slug}"] = pred
    df_re[f"confianza_{slug}"] = conf

slug_principal = modelo_principal_nombre.lower().replace(" ", "_").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
df_re["clasificacion_modelo_principal"] = df_re[f"clasificacion_{slug_principal}"]
df_re["confianza_modelo_principal"] = df_re[f"confianza_{slug_principal}"]
df_re["umbral_academico"] = umbral_confianza
df_re["confianza_suficiente"] = df_re["confianza_modelo_principal"] >= umbral_confianza

# RoBERTuito como benchmark externo.
if analyzer is not None:
    textos_robertuito = df_re["comentario_limpio"].head(int(limite_comentarios_robertuito))
    resultados_robertuito = textos_robertuito.apply(lambda x: analizar_sentimiento_robertuito(analyzer, x))
    df_re["clasificacion_robertuito"] = "No procesado"
    df_re["confianza_robertuito"] = np.nan
    df_re.loc[textos_robertuito.index, "clasificacion_robertuito"] = [r[0] for r in resultados_robertuito]
    df_re.loc[textos_robertuito.index, "confianza_robertuito"] = [r[1] for r in resultados_robertuito]
else:
    df_re["clasificacion_robertuito"] = "No disponible"
    df_re["confianza_robertuito"] = np.nan

df_re["concordancia_modelos"] = np.where(
    df_re["clasificacion_robertuito"].isin(["No disponible", "No procesado"]),
    "No disponible",
    np.where(df_re["clasificacion_modelo_principal"] == df_re["clasificacion_robertuito"], "Coincide", "No coincide")
)
df_re["senal_critica_detectada"] = df_re["comentario_limpio"].apply(detectar_senal_critica)
df_re["conector_contraste_detectado"] = df_re["comentario_limpio"].apply(detectar_contraste)
df_re["temas_pedagogicos"] = df_re["comentario_limpio"].apply(detectar_temas)
df_re["decision_final_sistema"] = df_re.apply(lambda row: construir_decision_final(row, umbral_confianza), axis=1)
df_re["prioridad_revision"] = df_re.apply(clasificar_intensidad_revision, axis=1)

# Columnas de validación experta por comentario.
df_re["validacion_experto"] = "Pendiente"
df_re["comentario_experto"] = ""

# ============================================================
# RESÚMENES Y MÉTRICAS DE PANTALLA
# ============================================================
nota_texto = "No disponible" if pd.isna(nota_global) else f"{nota_global:.2f}/5"
porc_pos = df_re["clasificacion_modelo_principal"].eq("Positivo").mean() * 100
porc_neu = df_re["clasificacion_modelo_principal"].eq("Neutro").mean() * 100
porc_neg = df_re["clasificacion_modelo_principal"].eq("Negativo").mean() * 100
porc_revision = df_re["decision_final_sistema"].isin(["Revisión académica", "Revisión por baja confianza", "Negativo confirmado"]).mean() * 100
porc_baja_confianza = (~df_re["confianza_suficiente"]).mean() * 100
conf_media_modelo = df_re["confianza_modelo_principal"].mean() * 100

mask_robertuito_util = df_re["clasificacion_robertuito"].isin(["Positivo", "Neutro", "Negativo"])
if mask_robertuito_util.any():
    concordancia = (df_re.loc[mask_robertuito_util, "clasificacion_modelo_principal"] == df_re.loc[mask_robertuito_util, "clasificacion_robertuito"]).mean() * 100
    conf_media_robertuito = df_re.loc[mask_robertuito_util, "confianza_robertuito"].mean() * 100
else:
    concordancia = np.nan
    conf_media_robertuito = np.nan

if porc_revision >= 30 or porc_neg >= 30:
    nivel_alerta = "Alto"
    interpretacion = "Se recomienda revisión académica prioritaria, análisis del director/jefe de área y definición de un plan de acompañamiento docente."
elif porc_revision >= 15 or porc_neg >= 15 or porc_baja_confianza >= 30:
    nivel_alerta = "Medio"
    interpretacion = "Se recomienda seguimiento pedagógico, revisión de comentarios prioritarios y verificación experta de las señales detectadas."
else:
    nivel_alerta = "Bajo"
    interpretacion = "Se recomienda seguimiento regular, mantenimiento de buenas prácticas y revisión puntual de sugerencias de mejora."

df_temas = resumen_temas(df_re)
texto_total = " ".join(df_re["comentario_limpio"].tolist())
palabras = [p for p in texto_total.split() if p not in PALABRAS_EXCLUIR and len(p) > 3]
df_palabras = pd.DataFrame(Counter(palabras).most_common(20), columns=["Palabra", "Frecuencia"])

# Tabla comparativa de confianza por modelo.
resumen_modelos = []
for nombre_modelo in modelos.keys():
    slug = nombre_modelo.lower().replace(" ", "_").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    col_pred = f"clasificacion_{slug}"
    col_conf = f"confianza_{slug}"
    dist = calcular_distribucion(df_re, col_pred, nombre_modelo)
    resumen_modelos.append({
        "Modelo": nombre_modelo,
        "Modelo ganador global": "Sí" if nombre_modelo == modelo_principal_nombre else "No",
        "Confianza promedio": df_re[col_conf].mean(),
        "Umbral académico": umbral_confianza,
        "% bajo umbral": (df_re[col_conf] < umbral_confianza).mean() * 100,
        "% Positivo": dist[f"% {nombre_modelo} Positivo"],
        "% Neutro": dist[f"% {nombre_modelo} Neutro"],
        "% Negativo": dist[f"% {nombre_modelo} Negativo"],
    })
df_resumen_modelos = pd.DataFrame(resumen_modelos)

# Comentarios prioritarios.
df_alerta = df_re[
    df_re["decision_final_sistema"].isin(["Revisión académica", "Revisión por baja confianza", "Negativo confirmado"])
    | df_re["senal_critica_detectada"]
    | df_re["conector_contraste_detectado"]
    | df_re["temas_pedagogicos"].apply(len).gt(0)
].copy()

resumen_para_reportes = {
    "comentarios_validos": comentarios_validos,
    "modelo_principal": modelo_principal_nombre,
    "umbral": umbral_confianza,
    "confianza_modelo": conf_media_modelo,
    "concordancia": 0 if pd.isna(concordancia) else concordancia,
    "porc_revision": porc_revision,
    "nivel_alerta": nivel_alerta,
    "porc_pos": porc_pos,
    "porc_neu": porc_neu,
    "porc_neg": porc_neg,
}

ruta_1 = "Revisar comentarios priorizados, mantener seguimiento preventivo y reforzar los temas pedagógicos más frecuentes en el siguiente módulo."
ruta_2 = "Cuando existan comentarios negativos confirmados o baja confianza, solicitar revisión del director/jefe de área y acordar un plan de mejora con evidencia verificable."
recomendaciones_basicas = generar_recomendaciones_basicas(df_temas, nivel_alerta)

# ============================================================
# VISUALIZACIÓN: ORDEN FINAL DE STREAMLIT
# ============================================================
st.markdown("## 1. Resumen ejecutivo para Dirección")
st.write(
    "Esta primera vista concentra las analíticas principales. El usuario principal es el director, jefe de área o responsable académico. "
    "Los comentarios detallados aparecen más abajo para evitar saturar la lectura inicial."
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Nota Likert", nota_texto)
k2.metric("Comentarios válidos", f"{comentarios_validos:,}")
k3.metric("Percepción positiva", f"{porc_pos:.1f}%")
k4.metric("Percepción negativa", f"{porc_neg:.1f}%")
k5.metric("Requiere revisión", f"{porc_revision:.1f}%")
k6.metric("Alerta", nivel_alerta)

st.info(
    f"Modelo principal/global: **{modelo_principal_nombre}**. "
    f"Umbral académico aplicado: **{umbral_confianza:.2f}**. "
    "La decisión final del sistema prioriza revisión experta cuando hay baja confianza, discrepancias con RoBERTuito o señales críticas."
)

st.markdown("## 2. Analítica cuantitativa y distribución general")
tab_likert, tab_percepcion, tab_modelos = st.tabs(["Likert", "Clasificador de tres clases", "Comparación de modelos"])

with tab_likert:
    st.write("La escala Likert resume la valoración cuantitativa antes de analizar comentarios abiertos.")
    if not df_likert_resumen.empty:
        st.dataframe(df_likert_resumen, use_container_width=True)
        st.pyplot(graficar_barras(df_likert_resumen, "Categoría Likert", "Frecuencia", "Distribución de respuestas Likert"))
    else:
        st.warning("No se encontraron respuestas Likert válidas.")

with tab_percepcion:
    df_dist = df_re["clasificacion_modelo_principal"].value_counts().reindex(["Positivo", "Neutro", "Negativo"]).fillna(0).reset_index()
    df_dist.columns = ["Clasificación", "Cantidad"]
    st.dataframe(df_dist, use_container_width=True)
    st.pyplot(graficar_barras(df_dist, "Clasificación", "Cantidad", "Distribución del clasificador de tres grandes categorías"))
    st.write(
        "El sistema clasifica los comentarios en tres categorías principales: **Positivo**, **Neutro** y **Negativo**. "
        "Los comentarios neutros pueden contener sugerencias de mejora sin representar una queja directa."
    )

with tab_modelos:
    st.write(
        "Esta tabla separa el **modelo ganador global** de la **confianza por comentario**. "
        "Un modelo puede mostrar alta confianza en ciertos comentarios, pero el ganador global se define por desempeño experimental o por el modelo exportado desde el notebook."
    )
    df_resumen_vista = df_resumen_modelos.copy()
    df_resumen_vista["Confianza promedio"] = (df_resumen_vista["Confianza promedio"] * 100).round(1).astype(str) + "%"
    df_resumen_vista["Umbral académico"] = df_resumen_vista["Umbral académico"].round(2)
    df_resumen_vista["% bajo umbral"] = df_resumen_vista["% bajo umbral"].round(1).astype(str) + "%"
    for c in ["% Positivo", "% Neutro", "% Negativo"]:
        df_resumen_vista[c] = df_resumen_vista[c].round(1).astype(str) + "%"
    st.dataframe(df_resumen_vista, use_container_width=True)

    if metricas_modelos is not None and isinstance(metricas_modelos, pd.DataFrame) and not metricas_modelos.empty:
        st.write("### Métricas experimentales cargadas desde el entrenamiento")
        st.dataframe(metricas_modelos, use_container_width=True)
    else:
        st.warning(
            "No se encontró un archivo de métricas experimentales. Para defensa, exporta desde el notebook una tabla como "
            "metricas_modelos.csv con Accuracy, Precision, Recall y F1 macro de validación/test."
        )

st.markdown("## 3. Confianza, umbral académico y validación de resultados")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Confianza promedio modelo", f"{conf_media_modelo:.1f}%")
col_b.metric("Bajo umbral", f"{porc_baja_confianza:.1f}%")
col_c.metric("Concordancia con RoBERTuito", "No disponible" if pd.isna(concordancia) else f"{concordancia:.1f}%")
col_d.metric("Confianza RoBERTuito", "No disponible" if pd.isna(conf_media_robertuito) else f"{conf_media_robertuito:.1f}%")

st.write(
    "El umbral académico no afirma que el modelo tenga la verdad final. Indica cuándo una predicción tiene confianza suficiente "
    "para ser leída como apoyo analítico y cuándo debe pasar a revisión humana."
)

st.markdown("## 4. Temas pedagógicos, palabras frecuentes y tendencias")
tab_temas, tab_palabras = st.tabs(["Temas pedagógicos", "Palabras frecuentes"])
with tab_temas:
    if not df_temas.empty:
        st.dataframe(df_temas, use_container_width=True)
        st.pyplot(graficar_barras(df_temas.head(10), "Frecuencia", "Tema pedagógico", "Temas pedagógicos detectados", horizontal=True))
    else:
        st.warning("No se detectaron temas pedagógicos con las reglas actuales.")

with tab_palabras:
    if not df_palabras.empty:
        st.dataframe(df_palabras, use_container_width=True)
        st.pyplot(graficar_barras(df_palabras.head(15), "Frecuencia", "Palabra", "Top palabras relevantes", horizontal=True))
    else:
        st.warning("No se encontraron palabras relevantes suficientes después de la limpieza.")

st.markdown("## 5. Comentarios y predicciones detalladas")
st.write(
    "A partir de esta sección se muestran los comentarios. Las columnas relacionadas están juntas para facilitar la revisión: "
    "modelo principal, confianza, umbral, RoBERTuito, concordancia, señales críticas, temas y decisión final."
)

columnas_modelos_detalle = []
for nombre_modelo in modelos.keys():
    slug = nombre_modelo.lower().replace(" ", "_").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    columnas_modelos_detalle.extend([f"clasificacion_{slug}", f"confianza_{slug}"])

columnas_detalle = [
    "Comentario original",
    "clasificacion_modelo_principal",
    "confianza_modelo_principal",
    "umbral_academico",
    "confianza_suficiente",
    "clasificacion_robertuito",
    "confianza_robertuito",
    "concordancia_modelos",
    "senal_critica_detectada",
    "conector_contraste_detectado",
    "temas_pedagogicos",
    "decision_final_sistema",
    "prioridad_revision",
    "validacion_experto",
    "comentario_experto",
]

vista_comentarios = df_re[columnas_detalle].copy()
vista_comentarios = vista_comentarios.rename(columns={
    "clasificacion_modelo_principal": "Clasificación modelo principal",
    "confianza_modelo_principal": "Confianza modelo principal",
    "umbral_academico": "Umbral académico",
    "confianza_suficiente": "Confianza suficiente",
    "clasificacion_robertuito": "Clasificación RoBERTuito",
    "confianza_robertuito": "Confianza RoBERTuito",
    "concordancia_modelos": "Concordancia modelos",
    "senal_critica_detectada": "Señal crítica detectada",
    "conector_contraste_detectado": "Conector contraste detectado",
    "temas_pedagogicos": "Temas pedagógicos",
    "decision_final_sistema": "Decisión final del sistema",
    "prioridad_revision": "Prioridad revisión",
    "validacion_experto": "Validación experto",
    "comentario_experto": "Comentario experto",
})

# Mostrar primero comentarios priorizados.
filtro = st.radio(
    "Vista de comentarios",
    ["Prioritarios", "Todos"],
    horizontal=True
)
if filtro == "Prioritarios":
    indices = df_alerta.index
    vista_para_editor = vista_comentarios.loc[indices].copy()
else:
    vista_para_editor = vista_comentarios.copy()

if hasattr(st, "column_config"):
    edited = st.data_editor(
        vista_para_editor,
        use_container_width=True,
        height=430,
        column_config={
            "Validación experto": st.column_config.SelectboxColumn(
                "Validación experto",
                options=["Pendiente", "Estoy de acuerdo", "No estoy de acuerdo", "Requiere segunda revisión"],
                required=True
            ),
            "Comentario experto": st.column_config.TextColumn("Comentario experto"),
        },
        disabled=[c for c in vista_para_editor.columns if c not in ["Validación experto", "Comentario experto"]]
    )
else:
    edited = st.data_editor(vista_para_editor, use_container_width=True, height=430)

st.markdown("## 6. Validación global del experto")
col_v1, col_v2 = st.columns([1, 2])
with col_v1:
    validacion_global = st.radio(
        "Después de revisar el informe:",
        ["Recibí el informe, lo leí y estoy de acuerdo", "No estoy de acuerdo", "Requiere segunda revisión"],
        index=0
    )
with col_v2:
    comentario_global_experto = st.text_area(
        "Comentarios finales del evaluador académico",
        placeholder="Ejemplo: Revisé los comentarios priorizados, confirmé las señales críticas y recomiendo seguimiento pedagógico en retroalimentación y claridad de rúbricas."
    )

st.markdown("## 7. Recomendaciones con IA y dos soluciones de acción")
st.write(
    "Esta sección va después de la analítica y los comentarios. Su función es convertir los datos en recomendaciones para el director y para el docente."
)

prompt_base = f"""
Actúa como experto en evaluación docente, analítica educativa y mejora académica.

Contexto:
- Usuario principal de la herramienta: director, jefe de área o responsable académico.
- Propósito: apoyar decisiones y priorizar revisión humana, no reemplazar al experto.
- Modelo principal: {modelo_principal_nombre} con representación NLP TF-IDF entrenado con etiquetas humanas.
- Benchmark externo: RoBERTuito.

Resultados:
- Nota Likert: {nota_texto}
- Comentarios válidos: {comentarios_validos}
- Comentarios eliminados: {comentarios_eliminados}
- Percepción positiva: {porc_pos:.1f}%
- Percepción neutra: {porc_neu:.1f}%
- Percepción negativa: {porc_neg:.1f}%
- Umbral académico: {umbral_confianza:.2f}
- Comentarios bajo umbral: {porc_baja_confianza:.1f}%
- Comentarios para revisión: {porc_revision:.1f}%
- Confianza promedio modelo principal: {conf_media_modelo:.1f}%
- Concordancia con RoBERTuito: {'No disponible' if pd.isna(concordancia) else f'{concordancia:.1f}%'}
- Nivel de alerta: {nivel_alerta}
- Temas pedagógicos: {df_temas.head(10).to_dict(orient='records') if not df_temas.empty else []}
- Palabras frecuentes: {df_palabras.head(10).to_dict(orient='records') if not df_palabras.empty else []}
- Comentarios prioritarios: {df_alerta['Comentario original'].head(8).tolist() if not df_alerta.empty else []}

Genera dos salidas diferenciadas:
1. Informe para director/jefe de área: resumen ejecutivo, riesgo académico, evidencia y dos rutas de acción.
2. Informe para docente: fortalezas, puntos de mejora y recomendaciones accionables en tono formativo.
Evita tono punitivo. Incluye una nota ética de revisión humana.
"""

respuesta_gemini = None
if GEMINI_API_KEY and GEMINI_INSTALADO:
    if st.button("Generar recomendaciones con Gemini"):
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            modelos_ia = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest", "gemini-pro-latest"]
            ultimo_error = None
            for nombre in modelos_ia:
                try:
                    model_ia = genai.GenerativeModel(nombre)
                    response = model_ia.generate_content(prompt_base)
                    respuesta_gemini = response.text
                    st.success(f"Recomendaciones generadas con {nombre}")
                    st.write(respuesta_gemini)
                    break
                except Exception as e:
                    ultimo_error = e
            if respuesta_gemini is None:
                st.error(f"No se pudo generar con Gemini. Último error: {ultimo_error}")
        except Exception as e:
            st.error(f"Error general con Gemini: {e}")
else:
    st.info("Gemini no está configurado. Se muestran recomendaciones base generadas por reglas de apoyo.")

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown("### Solución 1: Seguimiento preventivo")
    st.write(ruta_1)
with col_r2:
    st.markdown("### Solución 2: Plan de mejora formal")
    st.write(ruta_2)

st.markdown("### Recomendaciones base para el docente")
st.write(recomendaciones_basicas)

st.markdown("## 8. Reportes y descarga")
reporte_director = generar_reporte_director(nota_texto, resumen_para_reportes, df_temas, df_alerta, interpretacion, ruta_1, ruta_2)
reporte_docente = generar_reporte_docente(nota_texto, resumen_para_reportes, df_temas, recomendaciones_basicas)

col_d1, col_d2, col_d3 = st.columns(3)
with col_d1:
    st.download_button(
        "Descargar reporte para director",
        data=reporte_director.encode("utf-8"),
        file_name="reporte_director_desempeno_docente.md",
        mime="text/markdown"
    )
with col_d2:
    st.download_button(
        "Descargar reporte para docente",
        data=reporte_docente.encode("utf-8"),
        file_name="reporte_docente_mejora_academica.md",
        mime="text/markdown"
    )
with col_d3:
    st.download_button(
        "Descargar base completa CSV",
        data=df_re.to_csv(index=False, encoding="utf-8-sig"),
        file_name="resultados_detallados_desempeno_docente.csv",
        mime="text/csv"
    )

st.markdown("## 9. Preparación de envío")
st.write(
    "Para automatizar envío real se recomienda usar credenciales institucionales y autorización del área correspondiente. "
    "Esta versión deja preparados los reportes y plantillas de correo para evitar envíos automáticos sin revisión humana."
)

correo_director = st.text_input("Correo del director / jefe de área", placeholder="director@udla.edu.ec")
correo_docente = st.text_input("Correo del docente", placeholder="docente@udla.edu.ec")

asunto_director = "Reporte de análisis de desempeño docente"
cuerpo_director = "Estimado/a, adjunto o comparto el reporte de análisis de desempeño docente para revisión académica.\n\n" + reporte_director[:1500]
asunto_docente = "Retroalimentación formativa de evaluación docente"
cuerpo_docente = "Estimado/a docente, comparto el resumen formativo de retroalimentación para mejora continua.\n\n" + reporte_docente[:1500]

if correo_director:
    mailto_director = f"mailto:{correo_director}?subject={urllib.parse.quote(asunto_director)}&body={urllib.parse.quote(cuerpo_director)}"
    st.markdown(f"[Preparar correo para director]({mailto_director})")
if correo_docente:
    mailto_docente = f"mailto:{correo_docente}?subject={urllib.parse.quote(asunto_docente)}&body={urllib.parse.quote(cuerpo_docente)}"
    st.markdown(f"[Preparar correo para docente]({mailto_docente})")

# Exportar validación global del experto.
validacion_df = pd.DataFrame([{
    "fecha_revision": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "validacion_global": validacion_global,
    "comentario_global_experto": comentario_global_experto,
    "modelo_principal": modelo_principal_nombre,
    "umbral_academico": umbral_confianza,
    "nivel_alerta": nivel_alerta,
}])
st.download_button(
    "Descargar validación del experto CSV",
    data=validacion_df.to_csv(index=False, encoding="utf-8-sig"),
    file_name="validacion_experto.csv",
    mime="text/csv"
)

with st.expander("Ver archivo original cargado"):
    st.dataframe(df_raw.head(30), use_container_width=True)
