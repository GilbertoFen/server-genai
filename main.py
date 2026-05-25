from fastapi import FastAPI, HTTPException  # Importamos HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
# Importante para configurar el formato JSON de salida
from google.genai import types
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

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


class ExperienceRequest(BaseModel):
    experience_text: str

class AnalysisRequest(BaseModel):
    student_context: str
    
class ChatRequest(BaseModel):
    message: str
    student_profile: str  # Aquí NestJS mandará: "Alumno: Gil, Promedio: 9, Intereses: Cloud..."
    history: List[Dict[str, Any]]

@app.get("/")
def root():
    return {"status": "API de PumaIA funcionando 🚀"}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # 2. INYECTAMOS EL PERFIL COMO INSTRUCCIÓN DEL SISTEMA (Contexto)
        instruccion_sistema = f"""
        Eres PumaIA, un asistente académico experto de la carrera MAC en FES Acatlán.
        ESTE ES EL PERFIL ACTUALIZADO DEL ALUMNO CON EL QUE ESTÁS HABLANDO:
        {req.student_profile}
        
        Usa esta información para dar respuestas personalizadas. Si te pregunta su promedio o intereses, dáselos basados en este contexto.
        """

        # 3. RECONSTRUIMOS LA MEMORIA DE LA CONVERSACIÓN
        # Transformamos el historial que manda NestJS al formato que pide la API de Google
        mensajes_historial = []
        for msg in req.history:
            # En Gemini, el rol del bot se llama "model", el usuario es "user"
            rol_gemini = "model" if msg["role"] == "ASSISTANT" else "user"
            mensajes_historial.append({
                "role": rol_gemini,
                "parts": [{"text": msg["content"]}]
            })
        
        # Agregamos el mensaje nuevo del usuario al final de la historia
        mensajes_historial.append({
            "role": "user",
            "parts": [{"text": req.message}]
        })

        # 4. LLAMAMOS A GEMINI CON TODO EL CONTEXTO Y LA MEMORIA
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=mensajes_historial, # Mandamos la historia completa + el mensaje nuevo
            config={
                "system_instruction": instruccion_sistema # Le damos el perfil del alumno como regla base
            }
        )

        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-experience")
async def analyze_experience(req: ExperienceRequest):
    try:
        # Armamos el prompt estricto
        prompt = f"""
        Eres un reclutador experto en TI. Analiza la siguiente experiencia profesional:
        "{req.experience_text}"
        
        Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura, sin bloques de código Markdown (```json) ni texto adicional:
        {{
          "areaExpertise": "Nombre corto del área, ej. Desarrollo Backend",
          "categoryEnum": "DEBE SER EXACTAMENTE UNO DE ESTOS VALORES: CIENCIAS_DE_LA_COMPUTACION, SISTEMAS_COMPUTACIONALES, ADMINISTRACION_Y_FINANZAS, MATEMATICAS, MATEMATICAS_COMPUTACIONALES"
        }}
        """

        # Usamos tu sintaxis actual del cliente de Google
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", # Usa el que ya tienes
            contents=prompt,
            # Si tu versión del SDK lo permite, fuerza el JSON así:
            # config={"response_mime_type": "application/json"}
        )

        # Gemini a veces devuelve el string con ```json ... ```, lo limpiamos por si acaso
        raw_text = response.text.replace("```json", "").replace("```", "").strip()
        
        # Lo convertimos a un diccionario de Python para que FastAPI lo envíe como JSON real a NestJS
        parsed_json = json.loads(raw_text)

        return parsed_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en IA: {str(e)}")

@app.post("/analyze-profile")
async def analyze_profile(req: AnalysisRequest):
    try:
        prompt_maestro = f"""
        Eres PumaIA, un modelo experto en orientación profesional y laboral para alumnos de la carrera de Matemáticas Aplicadas y Computación (MAC) de la FES Acatlán, UNAM.
        
        A continuación tienes el contexto completo de un alumno (materias, intereses, cursos, promedio):
        {req.student_context}
        
        Tu tarea es generar un informe detallado basado estrictamente en sus datos que contenga:
        1. Resumen General del perfil.
        2. Fortalezas detectadas.
        3. Áreas de oportunidad.
        4. Exactamente 5 rutas profesionales recomendadas (Opción A, B, C, D, E) con su porcentaje de compatibilidad y recomendaciones específicas de qué tecnologías aprender o qué materias optativas cursar en la FES Acatlán.
        
        CRITERIO ESTRICTO DE PONDERACIÓN PARA EL PORCENTAJE DE COMPATIBILIDAD (matchPercent):
        Calcula el porcentaje de compatibilidad (un número entero del 0 al 100) basándote en la siguiente regla:
        - 90% a 100%: Solo si el alumno YA TIENE experiencia práctica demostrada (freelance, proyectos, trabajo), cursos acreditados O un historial de materias aprobadas directamente relacionadas con esa área.
        - 75% a 89%: Si el alumno tiene un interés fuerte y bases matemáticas/lógicas sólidas de MAC que faciliten su aprendizaje, pero carece de proyectos o experiencia práctica en esa área específica.
        - 50% a 74%: Si es una ruta viable gracias a su carrera de MAC, pero el alumno no ha manifestado interés ni tiene experiencia previa en ella.
        - NUNCA uses los mismos porcentajes repetitivos para todos los alumnos. Evalúa con criterio real y riguroso.

        RESTRICCIÓN CRÍTICA DE FORMATO:
        Debes responder EXCLUSIVAMENTE con un objeto JSON válido que contenga la siguiente estructura exacta. No agregues texto introductorio, ni bloques de código tipo markdown (sin ```json):
        {{
          "optionA": "Título de la Ruta 1",
            "descriptionA": "Match: [Inserta el porcentaje calculado]%. Resumen: ... Recomendaciones: ...",
            "optionB": "Título de la Ruta 2",
            "descriptionB": "Match: [Inserta el porcentaje calculado]%. Resumen: ... Recomendaciones: ...",
            "optionC": "Título de la Ruta 3",
            "descriptionC": "Match: [Inserta el porcentaje calculado]%. Resumen: ... Recomendaciones: ...",
            "optionD": "Título de la Ruta 4",
            "descriptionD": "Match: [Inserta el porcentaje calculado]%. Resumen: ... Recomendaciones: ...",
            "optionE": "Título de la Ruta 5",
            "descriptionE": "Match: [Inserta el porcentaje calculado]%. Resumen: ... Recomendaciones: ...",
          "meta_summary": "Escribe aquí el resumen general del perfil que irá en el recuadro dorado superior.",
          "meta_strengths": "Escribe aquí las fortalezas encontradas, separadas por viñetas.",
          "meta_opportunities": "Escribe aquí las áreas de oportunidad encontradas, separadas por viñetas."
        }}
        """

        # Forzamos a Gemini a responder en un formato JSON estructurado rígido
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt_maestro,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        # Parseamos la respuesta para asegurar la validez antes de enviarla a NestJS
        result_json = json.loads(response.text)
        return result_json

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502, detail="La IA no generó una estructura JSON limpia.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
