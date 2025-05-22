import requests
import json
from utils.text_utils import wrap_text

def generate_response(prompt):
    respuesta = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gemma3", "prompt": prompt},
        stream=True
    )
    respuesta_completa = ""
    for linea in respuesta.iter_lines():
        if linea:
            parte = json.loads(linea)
            respuesta_completa += parte.get("response", "")
    return respuesta_completa.strip()

def ayuda_mode(question):
    prompt = f"""Responde a la siguiente pregunta de forma clara y sencilla, en español. Si no sabes la respuesta, dilo honestamente.

Pregunta: {question}
Respuesta:"""
    print("Ayuda LLaMA:\n" + wrap_text(generate_response(prompt)))

def generar_resumen(config, table_config):
    separator = table_config.get('separator', 'no especificado')
    missing_data = table_config.get('missing_value_strategy', 'no especificado')

    entrada = ", ".join([f['name'] for f in config['input_features']])
    salida = ", ".join([f['name'] for f in config['output_features']])
    prompt = (
        f"¿Cuál es el propósito de este modelo?\n"
        f"- Entradas: {entrada}\n"
        f"- Salida: {salida}\n"
        f"- Separador: {separator}\n"
        f"- Tratamiento de valores faltantes: {missing_data}\n\n"
        f"Respuesta (solo un párrafo en español, sin repetir la pregunta):"
    )
    return generate_response(prompt).strip()
