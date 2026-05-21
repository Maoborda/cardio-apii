
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, scipy.signal as sg, json, os

app = FastAPI(title="Cardio-IA v4")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

with open("/content/nombres_clases.json") as f:
    NC = json.load(f)

MODELO=None; META={"frecuencia_muestreo":100.0,"longitud_ventana":1000,"n_derivaciones":12,"umbral_confianza":0.65}

try:
    import tensorflow as tf
    if os.path.exists("/content/cardio_brain_24clases_v4.keras"):
        MODELO=tf.keras.models.load_model("/content/cardio_brain_24clases_v4.keras")
        print("Modelo v4 OK")
    if os.path.exists("/content/modelo_metadatos_v4.json"):
        with open("/content/modelo_metadatos_v4.json") as f:
            META=json.load(f)
except Exception as e:
    print(f"Error modelo: {e}")

def fil(s,fs=100.0):
    n=0.5*fs; b,a=sg.butter(4,[0.5/n,45.0/n],btype="band")
    f=sg.filtfilt(b,a,s); b2,a2=sg.iirnotch(50.0,Q=30.0,fs=fs)
    return sg.filtfilt(b2,a2,f)

def diag(p,u=0.65):
    i=int(np.argmax(p)); c=float(p[i])
    return {"diagnostico":"INDETERMINADO" if c<u else NC[i],"confianza":round(c*100,1),
            "requiere_revision":c<u,"distribucion":{n:round(float(v)*100,2) for n,v in zip(NC,p)}}

@app.get("/")
def r(): return {"mensaje":"Cardio-IA v4 — 24 clases","modelo_cargado":MODELO is not None}

@app.get("/salud")
def s(): return {"estado":"ok","modelo_cargado":MODELO is not None,"version":"4.0","n_clases":24,"clases":NC}

class E(BaseModel):
    senal: list
    frecuencia_muestreo: float = 100.0

@app.post("/predecir")
def pred(e: E):
    if MODELO is None:
        return {**diag(np.array([0.82]+[0.01]*23)),"modo":"DEMO"}
    try:
        s=np.array(e.senal,dtype=np.float32)
        L=META.get("longitud_ventana",1000); NL=META.get("n_derivaciones",12)
        if s.ndim==1: s=np.tile(s[:L,np.newaxis],(1,NL))
        s=s[:L]
        for i in range(s.shape[1]):
            c=fil(s[:,i],e.frecuencia_muestreo); std=np.std(c)
            s[:,i]=(c-np.mean(c))/(std+1e-8) if std>1e-8 else c
        p=MODELO.predict(s[np.newaxis].astype(np.float32),verbose=0)[0]
        return diag(p,META.get("umbral_confianza",0.65))
    except Exception as ex:
        raise HTTPException(status_code=500,detail=str(ex))

@app.post("/predecir-archivo")
async def parch(archivo: UploadFile = File(...)):
    c=await archivo.read()
    s=np.array([float(l.strip()) for l in c.decode().split("\n") if l.strip()],dtype=np.float32)
    if MODELO is None:
        res={**diag(np.array([0.82]+[0.01]*23)),"modo":"DEMO"}
    else:
        L=META.get("longitud_ventana",1000); NL=META.get("n_derivaciones",12)
        seg=np.tile(s[:L,np.newaxis],(1,NL))
        for i in range(seg.shape[1]):
            cf=fil(seg[:,i]); std=np.std(cf)
            seg[:,i]=(cf-np.mean(cf))/(std+1e-8) if std>1e-8 else cf
        p=MODELO.predict(seg[np.newaxis].astype(np.float32),verbose=0)[0]
        res=diag(p)
    res["muestras"]=len(s); return res
