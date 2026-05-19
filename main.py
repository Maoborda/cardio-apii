"""
Cardio-IA Latam — API FastAPI
==============================
Servidor que recibe señales de ECG y devuelve diagnósticos.

Endpoints:
  GET  /          → Estado de la API
  GET  /salud     → Health check con info del modelo
  POST /predecir  → Recibe señal ECG, devuelve diagnóstico
  POST /predecir-archivo → Recibe archivo CSV con la señal
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import scipy.signal as sig
import tensorflow as tf
import json
import io
import os

# ---------------------------------------------------------------------------
# Configuración de la app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cardio-IA Latam API",
    description="API de diagnóstico cardiovascular con IA",
    version="2.0",
)

# CORS — permite que el frontend React se conecte a esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En producción: poner solo la URL del frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Cargar modelo al iniciar el servidor
# ---------------------------------------------------------------------------
MODELO        = None
METADATOS     = None
NOMBRES_CLASES = [
    "Normal",
    "Fibrilación Auricular",
    "Taquicardia Ventricular",
    "Bradicardia",
    "Bloqueo",
    "Isquemia",
]

@app.on_event("startup")
async def cargar_modelo():
    global MODELO, METADATOS
    ruta_modelo    = os.getenv("RUTA_MODELO",    "cardio_brain_mejor.keras")
    ruta_metadatos = os.getenv("RUTA_METADATOS", "modelo_metadatos.json")
    try:
        MODELO = tf.keras.models.load_model(ruta_modelo)
        print(f"✅ Modelo cargado: {ruta_modelo}")
    except Exception as e:
        print(f"⚠️  No se pudo cargar el modelo: {e}")
        print("    La API funcionará pero /predecir devolverá error.")
    try:
        with open(ruta_metadatos, "r") as f:
            METADATOS = json.load(f)
        print(f"✅ Metadatos cargados")
    except Exception:
        METADATOS = {
            "version": "2.0",
            "frecuencia_muestreo": 360.0,
            "longitud_ventana": 540,
            "umbral_confianza": 0.70,
        }

# ---------------------------------------------------------------------------
# Funciones de procesamiento (mismo algoritmo robusto del notebook)
# ---------------------------------------------------------------------------
def limpiar_ecg(señal_raw: np.ndarray, fs: float = 360.0) -> np.ndarray:
    """Filtro Butterworth orden 4 + Notch 60 Hz"""
    if len(señal_raw) < 10:
        raise ValueError("Señal demasiado corta")
    if np.std(señal_raw) < 1e-5:
        raise ValueError("Señal plana — verifique electrodos")
    if np.any(np.isnan(señal_raw)) or np.any(np.isinf(señal_raw)):
        raise ValueError("Señal contiene valores inválidos (NaN/Inf)")
    nyquist  = 0.5 * fs
    b, a     = sig.butter(4, [0.5 / nyquist, 45.0 / nyquist], btype="band")
    filtrada = sig.filtfilt(b, a, señal_raw)
    b2, a2   = sig.iirnotch(60.0, Q=30.0, fs=fs)
    return sig.filtfilt(b2, a2, filtrada)

def preparar_ventana(señal_limpia: np.ndarray, longitud: int) -> np.ndarray:
    """Toma los primeros `longitud` samples y normaliza por ventana"""
    if len(señal_limpia) < longitud:
        raise ValueError(f"Señal muy corta: {len(señal_limpia)} muestras (mínimo {longitud})")
    ventana = señal_limpia[:longitud]
    std     = np.std(ventana)
    if std < 1e-8:
        raise ValueError("Ventana plana después del filtrado")
    normalizada = (ventana - np.mean(ventana)) / (std + 1e-8)
    # Shape requerido por Conv1D: (1, longitud, 1)
    return normalizada.reshape(1, longitud, 1).astype(np.float32)

def diagnosticar(probs: np.ndarray, umbral: float = 0.70) -> dict:
    """Aplica umbral de confianza y construye la respuesta"""
    idx_max   = int(np.argmax(probs))
    confianza = float(probs[idx_max])
    requiere_revision = confianza < umbral

    return {
        "diagnostico": (
            "INDETERMINADO — requiere revisión por cardiólogo"
            if requiere_revision
            else NOMBRES_CLASES[idx_max]
        ),
        "confianza":          round(confianza * 100, 1),
        "requiere_revision":  requiere_revision,
        "distribucion": {
            nombre: round(float(p) * 100, 2)
            for nombre, p in zip(NOMBRES_CLASES, probs)
        },
    }

# ---------------------------------------------------------------------------
# Modelos de entrada/salida (Pydantic)
# ---------------------------------------------------------------------------
class EntradaECG(BaseModel):
    señal: list[float]
    frecuencia_muestreo: float = 360.0

    class Config:
        json_schema_extra = {
            "example": {
                "señal": [0.12, 0.15, 0.20, 0.35, 1.20, 0.80, 0.10, -0.05],
                "frecuencia_muestreo": 360.0,
            }
        }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def raiz():
    return {
        "mensaje":  "🫀 Cardio-IA Latam API funcionando",
        "version":  "2.0",
        "docs":     "/docs",
    }

@app.get("/salud", tags=["General"])
def salud():
    return {
        "estado":          "ok",
        "modelo_cargado":  MODELO is not None,
        "clases":          NOMBRES_CLASES,
        "umbral_confianza": METADATOS.get("umbral_confianza", 0.70) if METADATOS else 0.70,
    }

@app.post("/predecir", tags=["Diagnóstico"])
def predecir(entrada: EntradaECG):
    """
    Recibe un array de voltajes del ECG y devuelve el diagnóstico.

    - **señal**: lista de valores flotantes (voltaje del ECG)
    - **frecuencia_muestreo**: Hz del dispositivo (default 360.0)
    """
    if MODELO is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Verifica que cardio_brain_mejor.keras esté en el servidor.",
        )
    try:
        fs       = entrada.frecuencia_muestreo
        longitud = METADATOS.get("longitud_ventana", 540) if METADATOS else 540
        señal    = np.array(entrada.señal, dtype=np.float32)
        limpia   = limpiar_ecg(señal, fs=fs)
        ventana  = preparar_ventana(limpia, longitud)
        probs    = MODELO.predict(ventana, verbose=0)[0]
        return diagnosticar(probs, umbral=METADATOS.get("umbral_confianza", 0.70))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/predecir-archivo", tags=["Diagnóstico"])
async def predecir_archivo(archivo: UploadFile = File(...)):
    """
    Recibe un archivo CSV con una columna de voltajes del ECG.
    El archivo debe tener una sola columna numérica, sin encabezado.

    Ejemplo de contenido del CSV:
        0.12
        0.15
        0.20
        ...
    """
    if MODELO is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    if not archivo.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .csv")
    try:
        contenido = await archivo.read()
        texto     = contenido.decode("utf-8")
        señal     = np.array([
            float(linea.strip())
            for linea in texto.strip().split("\n")
            if linea.strip() and not linea.strip().startswith("#")
        ], dtype=np.float32)

        fs       = METADATOS.get("frecuencia_muestreo", 360.0) if METADATOS else 360.0
        longitud = METADATOS.get("longitud_ventana", 540)      if METADATOS else 540
        limpia   = limpiar_ecg(señal, fs=fs)
        ventana  = preparar_ventana(limpia, longitud)
        probs    = MODELO.predict(ventana, verbose=0)[0]

        resultado = diagnosticar(probs)
        resultado["archivo"]       = archivo.filename
        resultado["muestras_leidas"] = len(señal)
        return resultado

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")
