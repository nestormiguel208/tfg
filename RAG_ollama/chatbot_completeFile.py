import os
import argparse
import yaml
import pandas as pd
import textwrap
import requests
import json

from ludwig.automl import create_auto_config


# --- Comunicación directa con la API de Ollama ---
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

print("\nModelo gemma3 conectado con Ollama.")

def wrap_text(text, width=200):
    return "\n".join(textwrap.wrap(text, width))

# --- Ayuda por RAG (directamente usando el API de Ollama) ---
def ayuda_mode(question):
    prompt = f"""Responde a la siguiente pregunta de forma clara y sencilla, en español. Si no sabes la respuesta, dilo honestamente.

Pregunta: {question}
Respuesta:"""
    print("Ayuda LLaMA:\n" + wrap_text(generate_response(prompt)))

def ask_question(prompt, options=None):
    while True:
        print(f"\n{prompt}")
        if options:
            for i, option in enumerate(options, 1):
                print(f"  {i}. {option}")
            user_input = input("Selecciona una opción (número o 'ayuda: pregunta'): ")
            if user_input.lower().startswith("ayuda:"):
                pregunta = user_input.replace("ayuda:", "").strip()
                ayuda_mode(pregunta)
                continue
            if user_input.isdigit():
                idx = int(user_input)
                if 1 <= idx <= len(options):
                    return options[idx - 1].split(':')[0].strip()
            print("Entrada inválida.")
        else:
            user_input = input("Tú: ")
            if user_input.lower().startswith("ayuda:"):
                pregunta = user_input.replace("ayuda:", "").strip()
                ayuda_mode(pregunta)
                continue
            return user_input

# --- Generación de resumen por LLaMA ---
def generar_resumen(config, table_config):
    # Verifica si 'preprocessing' y las claves dentro de table_config existen
    separator = table_config.get('separator', 'no especificado')
    missing_data = table_config.get('missing_value_strategy', 'no especificado')  # Usamos 'missing_value_strategy' en lugar de 'missing_data'

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

# --- Función principal ---
def main():
    parser = argparse.ArgumentParser(description="Chatbot LLaMA para configuración AutoML")
    parser.add_argument("archivo", help="Ruta al archivo de datos (.csv, .xlsx, .arff)")
    args = parser.parse_args()
    file_path = args.archivo

    # --- Cargar archivo ---
    if file_path.endswith(".arff"):
        from scipy.io import arff
        data, _ = arff.loadarff(file_path)
        df = pd.DataFrame(data)
    elif file_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        print("Formato de archivo no soportado.")
        return

    print(f"\nColumnas detectadas: {list(df.columns)}")

    # --- Preguntar columna objetivo ---
    target = ask_question("¿Cuál es la columna objetivo (de salida)?", None)
    if target not in df.columns:
        print(f"Columna '{target}' no encontrada.")
        return

    # --- Crear configuración automática con Ludwig ---
    print("\nGenerando configuración automática con Ludwig AutoML...")
    config = create_auto_config(
        dataset=df,
        target=target,
        time_limit_s=600,  # Ajusta el tiempo si quieres una búsqueda más larga
    )

    # --- Añadir configuración extra de preprocesamiento ---
    separator_options = [
        "comma: Columnas separadas por comas.",
        "semicolon: Columnas separadas por punto y coma.",
        "backslash: Columnas separadas por barra invertida."
    ]
    missing_data_options = [
        "fill_with_const: Valor específico.",
        "fill_with_mode: Valor más frecuente.",
        "fill_with_mean: Media.",
        "fill_with_false: Valor falso.",
        "bfill: Valor siguiente.",
        "ffill: Valor anterior.",
        "drop_row: Eliminar fila."
    ]
    separator = ask_question("Separador de columnas:", separator_options)
    missing_data = ask_question("¿Cómo tratar los datos faltantes?", missing_data_options)

    config["preprocessing"] = {
        "separator": separator,
        "missing_value_strategy": missing_data  # Asegúrate de usar 'missing_value_strategy'
    }

    print("\nConfiguración generada automáticamente:")
    print(yaml.dump(config, allow_unicode=True))

    # Guardar el archivo YAML temporalmente
    with open("auto_config.yml", "w") as f:
        yaml.dump(config, f, allow_unicode=True)
    print("\nArchivo 'auto_config.yml' generado temporalmente.")

    # Preguntar si el usuario quiere modificar la configuración
    while True:
        edit = ask_question("¿Quieres modificar alguna columna o configuración? (sí/no)", ["sí", "no"])
        if edit == "no":
            break

        # Permitir modificar la configuración del archivo
        feature_role_options = ["input", "output", "ninguna"]
        input_type_options = [
            "binary", "number", "category", "text", "vector", "image",
            "audio", "timeseries", "date"
        ]
        target_type_options = [
            "binary", "number", "category", "bag", "set", "sequence",
            "text", "vector", "audio", "date", "h3", "image", "timeseries"
        ]

        # Convertimos listas a dicts para edición
        inputs = {f['name']: f for f in config.get("input_features", [])}
        outputs = {f['name']: f for f in config.get("output_features", [])}

        print("\nColumnas disponibles para modificación:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        sel = input("Selecciona el número de la columna que deseas modificar: ")
        if not sel.isdigit() or int(sel) < 1 or int(sel) > len(df.columns):
            print("Selección inválida.")
            continue
        col_name = df.columns[int(sel) - 1]

        role = ask_question(f"Uso para la columna '{col_name}':", feature_role_options)

        inputs.pop(col_name, None)
        outputs.pop(col_name, None)

        if role == "input":
            tipo = ask_question(f"Tipo de dato de '{col_name}':", input_type_options)
            inputs[col_name] = {"name": col_name, "type": tipo}
        elif role == "output":
            tipo = ask_question(f"Tipo de objetivo de '{col_name}':", target_type_options)
            outputs[col_name] = {"name": col_name, "type": tipo}
        else:
            print(f"Columna '{col_name}' será ignorada.")

        config["input_features"] = list(inputs.values())
        config["output_features"] = list(outputs.values())

    # Guardar el archivo YAML después de las modificaciones
    with open("auto_config.yml", "w") as f:
        yaml.dump(config, f, allow_unicode=True)
    print("\nArchivo 'auto_config.yml' guardado correctamente con las modificaciones finales.")

    print("\nResumen generado por LLaMA:")
    print(wrap_text(generar_resumen(config, config.get("preprocessing", {}))))

if __name__ == "__main__":
    main()

