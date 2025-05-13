import argparse
import yaml
from ludwig.utils.automl import auto_config

from utils import io_utils, text_utils, rag_utils
from utils.interaction import ask_question
from api.gemma3WithOllama import generar_resumen
from utils.text_utils import wrap_text

def main():
    parser = argparse.ArgumentParser(description="Chatbot LLaMA para configuración AutoML")
    parser.add_argument("archivo", help="Ruta al archivo de datos (.csv, .xlsx, .arff)")
    args = parser.parse_args()
    file_path = args.archivo

    # --- Configuración de búsqueda por contexto (RAG) ---
    vectordb = rag_utils.setup_vectordb()

    # --- Cargar archivo de datos ---
    try:
        df = io_utils.load_file(file_path)
    except ValueError as e:
        print(str(e))
        return

    print(f"\nColumnas detectadas: {list(df.columns)}")

    # --- Preguntas básicas ---
    separator = ask_question("Separador de columnas:", [
        "comma: Columnas separadas por comas.",
        "semicolon: Columnas separadas por punto y coma.",
        "backslash: Columnas separadas por barra invertida."
    ], vectordb)

    missing_data = ask_question("¿Cómo tratar los datos faltantes?", [
        "fill_with_const: Valor específico.",
        "fill_with_mode: Valor más frecuente.",
        "fill_with_mean: Media.",
        "fill_with_false: Valor falso.",
        "bfill: Valor siguiente.",
        "ffill: Valor anterior.",
        "drop_row: Eliminar fila."
    ], vectordb)

    # --- Columna objetivo ---
    target = ask_question("¿Cuál es la columna objetivo (de salida)?", None, vectordb)
    if target not in df.columns:
        print(f" Columna '{target}' no encontrada.")
        return

    # --- Configuración automática con Ludwig ---
    config = auto_config(dataset=df, target=target, time_limit_s=60, backend="local")
    config["preprocessing"] = {"separator": separator, "missing_value_strategy": missing_data}

    print("\n Configuración generada automáticamente:")
    print(yaml.dump(config, allow_unicode=True))

    # --- Permitir edición de columnas ---
    inputs = {f['name']: f for f in config.get("input_features", [])}
    outputs = {f['name']: f for f in config.get("output_features", [])}

    while True:
        edit = ask_question("¿Quieres modificar alguna columna? (sí/no)", ["sí", "no"])
        if edit == "no":
            break

        print("\n Columnas disponibles:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        sel = input("Selecciona el número de la columna que deseas modificar: ")
        if not sel.isdigit() or int(sel) < 1 or int(sel) > len(df.columns):
            print(" Selección inválida.")
            continue

        col_name = df.columns[int(sel) - 1]
        role = ask_question(f"Uso para la columna '{col_name}':", ["input", "output", "ninguna"], vectordb)

        inputs.pop(col_name, None)
        outputs.pop(col_name, None)

        if role == "input":
            tipo = ask_question(f"Tipo de dato de '{col_name}':", [
                "binary", "number", "category", "text", "vector", "image", "audio", "timeseries", "date"
            ], vectordb)
            inputs[col_name] = {"name": col_name, "type": tipo}
        elif role == "output":
            tipo = ask_question(f"Tipo de objetivo de '{col_name}':", [
                "binary", "number", "category", "bag", "set", "sequence",
                "text", "vector", "audio", "date", "h3", "image", "timeseries"
            ], vectordb)
            outputs[col_name] = {"name": col_name, "type": tipo}
        else:
            print(f" Columna '{col_name}' será ignorada.")

    config["input_features"] = list(inputs.values())
    config["output_features"] = list(outputs.values())

    # --- Guardar configuración ---
    io_utils.save_config(config)
    print("\n Archivo 'config.yml' guardado correctamente.")

    # --- Generar resumen por LLaMA ---
    print("\n Resumen generado por LLaMA:")
    print(wrap_text(generar_resumen(config, config.get("preprocessing", {}))))

if __name__ == "__main__":
    main()
