# Cardio-IA Latam — Backend API

## Archivos necesarios
Antes de hacer deploy, copia estos archivos a esta carpeta:
- `cardio_brain_mejor.keras`  (descárgalo de Google Drive/CardioIA/)
- `modelo_metadatos.json`     (descárgalo de Google Drive/CardioIA/)

## Correr localmente
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Abre http://localhost:8000/docs para ver la documentación interactiva.

## Deploy en Railway
1. Sube esta carpeta a un repositorio de GitHub
2. Entra a railway.app con tu cuenta de GitHub
3. Clic en "New Project" → "Deploy from GitHub repo"
4. Selecciona el repositorio
5. Railway detecta el Procfile y hace el deploy automáticamente
6. En 3 minutos tienes una URL pública tipo: https://cardio-ia.up.railway.app
