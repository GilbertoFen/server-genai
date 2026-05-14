from fastapi import FastAPI, HTTPException # Importamos HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# Inicializar cliente
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "API de PumaIA funcionando 🚀"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        contexto = """
        La carrera MAC en FES Acatlán incluye:
        - Álgebra Lineal
        - Ecuaciones Diferenciales
        - Programación
        """

        prompt = f"Eres PumaIA, asistente académico. Contexto: {contexto}. Usuario: {req.message}"

        # Usando el modelo especificado en tu código
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Nota: Asegúrate de que el nombre del modelo sea el correcto (ej. gemini-2.0-flash)
            contents=prompt
        )

        return {"response": response.text}

    except Exception as e:
        # Esto ayuda a que Render te muestre el error en los logs si algo falla
        raise HTTPException(status_code=500, detail=str(e))