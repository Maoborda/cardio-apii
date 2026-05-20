from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json, os

app = FastAPI(title="Cardio-IA Latam API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NOMBRES_CLASES = [
    "Normal",
    "Fibrilación Auricular",
    "Taquicardia Ventricular",
    "Bradicardia",
    "Bloqueo",
    "Isquemia",
]

def filtrar_ecg_simple(señal):
    """Filtro simple sin scipy — media móvil para eliminar ruido"""
    ventana = 5
    return np.convolve(señal, np.ones(ventana)/ventana, mode='same')

def diagnosticar_demo():
    """Respuesta demo cuando el modelo no está cargado"""
    probs = np.array([0.82, 0.05, 0.04, 0.03, 0.03, 0.03])
    idx   = int(np.argmax(probs))
    return {
        "diagnostico":       NOMBRES_CLASES[idx],
        "confianza":         82.0,
        "requiere_revision": False,
        "modo":              "DEMO",
        "distribucion": {
            n: round(float(p)*100, 2)
            for n, p in zip(NOMBRES_CLASES, probs)
        },
    }

@app.get("/")
def raiz():
    return {
        "mensaje": "🫀 Cardio-IA Latam API funcionando",
        "version": "2.0",
        "estado":  "ok",
        "docs":    "/docs",
        "salud":   "/salud",
    }

@app.get("/salud")
def salud():
    return {
        "estado":  "ok",
        "version": "2.0",
        "clases":  NOMBRES_CLASES,
    }

class EntradaECG(BaseModel):
    señal: list
    frecuencia_muestreo: float = 360.0

@app.post("/predecir")
def predecir(entrada: EntradaECG):
    try:
        señal   = np.array(entrada.señal, dtype=np.float32)
        filtrada = filtrar_ecg_simple(señal)
        return diagnosticar_demo()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predecir-archivo")
async def predecir_archivo(archivo: UploadFile = File(...)):
    if not archivo.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .csv")
    try:
        contenido = await archivo.read()
        señal = np.array([
            float(l.strip())
            for l in contenido.decode("utf-8").strip().split("\n")
            if l.strip()
        ], dtype=np.float32)
        res = diagnosticar_demo()
        res["archivo"]  = archivo.filename
        res["muestras"] = len(señal)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
