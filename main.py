from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import scipy.signal as sig
import json
import os

app = FastAPI(title="Cardio-IA Latam API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELO         = None
METADATOS      = None
NOMBRES_CLASES = [
    "Normal",
    "Fibrilación Auricular",
    "Taquicardia Ventricular",
    "Bradicardia",
    "Bloqueo",
    "Isquemia",
]

def cargar_modelo_al_inicio():
    global MODELO, METADATOS
    try:
        import tensorflow as tf
        ruta = os.getenv("RUTA_MODELO", "cardio_brain_mejor.keras")
        if os.path.exists(ruta):
            MODELO = tf.keras.models.load_model(ruta)
            print(f"✅ Modelo cargado: {ruta}")
        else:
            print(f"⚠️ Modelo no encontrado en: {ruta}")
    except Exception as e:
        print(f"⚠️ Error cargando modelo: {e}")

    try:
        ruta_meta = os.getenv("RUTA_METADATOS", "modelo_metadatos.json")
        if os.path.exists(ruta_meta):
            with open(ruta_meta, "r") as f:
                METADATOS = json.load(f)
        else:
            METADATOS = {
                "frecuencia_muestreo": 360.0,
                "longitud_ventana": 540,
                "umbral_confianza": 0.70,
            }
    except Exception as e:
        print(f"⚠️ Error cargando metadatos: {e}")
        METADATOS = {
            "frecuencia_muestreo": 360.0,
            "longitud_ventana": 540,
            "umbral_confianza": 0.70,
        }

cargar_modelo_al_inicio()

@app.get("/")
def raiz():
    return {
        "mensaje": "🫀 Cardio-IA Latam API funcionando",
        "version": "2.0",
        "docs": "/docs",
        "salud": "/salud",
    }

@app.get("/salud")
def salud():
    return {
        "estado": "ok",
        "modelo_cargado": MODELO is not None,
        "clases": NOMBRES_CLASES,
        "umbral_confianza": METADATOS.get("umbral_confianza", 0.70),
    }

def limpiar_ecg(señal_raw, fs=360.0):
    if len(señal_raw) < 10:
        raise ValueError("Señal demasiado corta")
    if np.std(señal_raw) < 1e-5:
        raise ValueError("Señal plana — verifique electrodos")
    nyquist  = 0.5 * fs
    b, a     = sig.butter(4, [0.5 / nyquist, 45.0 / nyquist], btype="band")
    filtrada = sig.filtfilt(b, a, señal_raw)
    b2, a2   = sig.iirnotch(60.0, Q=30.0, fs=fs)
    return sig.filtfilt(b2, a2, filtrada)

def preparar_ventana(señal_limpia, longitud):
    if len(señal_limpia) < longitud:
        raise ValueError(f"Señal muy corta: necesita al menos {longitud} muestras")
    ventana = señal_limpia[:longitud]
    std     = np.std(ventana)
    norm    = (ventana - np.mean(ventana)) / (std + 1e-8)
    return norm.reshape(1, longitud, 1).astype(np.float32)

def diagnosticar(probs, umbral=0.70):
    idx       = int(np.argmax(probs))
    confianza = float(probs[idx])
    return {
        "diagnostico": (
            "INDETERMINADO — requiere revisión por cardiólogo"
            if confianza < umbral
            else NOMBRES_CLASES[idx]
        ),
        "confianza": round(confianza * 100, 1),
        "requiere_revision": confianza < umbral,
        "distribucion": {
            n: round(float(p) * 100, 2)
            for n, p in zip(NOMBRES_CLASES, probs)
        },
    }

class EntradaECG(BaseModel):
    señal: list
    frecuencia_muestreo: float = 360.0

@app.post("/predecir")
def predecir(entrada: EntradaECG):
    if MODELO is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    try:
        señal   = np.array(entrada.señal, dtype=np.float32)
        fs      = entrada.frecuencia_muestreo
        long    = METADATOS.get("longitud_ventana", 540)
        limpia  = limpiar_ecg(señal, fs=fs)
        ventana = preparar_ventana(limpia, long)
        probs   = MODELO.predict(ventana, verbose=0)[0]
        return diagnosticar(probs, METADATOS.get("umbral_confianza", 0.70))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predecir-archivo")
async def predecir_archivo(archivo: UploadFile = File(...)):
    if MODELO is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    if not archivo.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .csv")
    try:
        contenido = await archivo.read()
        señal = np.array([
            float(l.strip())
            for l in contenido.decode("utf-8").strip().split("\n")
            if l.strip()
        ], dtype=np.float32)
        fs      = METADATOS.get("frecuencia_muestreo", 360.0)
        long    = METADATOS.get("longitud_ventana", 540)
        limpia  = limpiar_ecg(señal, fs=fs)
        ventana = preparar_ventana(limpia, long)
        probs   = MODELO.predict(ventana, verbose=0)[0]
        res     = diagnosticar(probs)
        res["archivo"] = archivo.filename
        res["muestras"] = len(señal)
        return res
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
