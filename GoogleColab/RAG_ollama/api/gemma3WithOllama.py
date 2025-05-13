import requests
import json

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

def generar_resumen(config, table_config):
    entrada = ", ".join([f['name'] for f in config['input_features']])
    salida = ", ".join([f['name'] for f in config['output_features']])
    prompt = (
        f"¿Cuál es el propósito de este modelo?\n"
        f"- Entradas: {entrada}\n"
        f"- Salida: {salida}\n"
        f"- Separador: {table_config['separator']}\n"
        f"- Tratamiento de valores faltantes: {table_config['missing_data']}\n\n"
        f"Respuesta (solo un párrafo en español, sin repetir la pregunta):"
    )
    return generate_response(prompt).strip()
