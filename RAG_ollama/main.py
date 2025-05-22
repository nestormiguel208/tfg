import argparse
import yaml
from ludwig.automl import create_auto_config
from utils import io_utils, interaction
from utils.text_utils import wrap_text
from api.gemma3WithOllama import generar_resumen

def main():
    parser = argparse.ArgumentParser(description="Chatbot LLaMA para configuración AutoML")
    parser.add_argument("archivo", help="Ruta al archivo de datos (.csv, .xlsx, .arff)")
    args = parser.parse_args()
    file_path = args.archivo

    df = io_utils.load_file(file_path)
    print(f"\nColumnas detectadas: {list(df.columns)}")

    target = interaction.ask_question("¿Cuál es la columna objetivo (de salida)?", None)
    if target not in df.columns:
        print(f"Columna '{target}' no encontrada.")
        return

    print("\nGenerando configuración automática con Ludwig AutoML...")
    config = create_auto_config(dataset=df, target=target, time_limit_s=600)

    separator = interaction.ask_question("Separador de columnas:", [
        "comma: Columnas separadas por comas.",
        "semicolon: Columnas separadas por punto y coma.",
        "backslash: Columnas separadas por barra invertida."
    ])
    missing_data = interaction.ask_question("¿Cómo tratar los datos faltantes?", [
        "fill_with_const: Valor específico.",
        "fill_with_mode: Valor más frecuente.",
        "fill_with_mean: Media.",
        "fill_with_false: Valor falso.",
        "bfill: Valor siguiente.",
        "ffill: Valor anterior.",
        "drop_row: Eliminar fila."
    ])
    config["preprocessing"] = {"separator": separator, "missing_value_strategy": missing_data}

    print("\nConfiguración generada automáticamente:")
    print(yaml.dump(config, allow_unicode=True))

    with open("auto_config.yml", "w") as f:
        yaml.dump(config, f, allow_unicode=True)

    while True:
        edit = interaction.ask_question("¿Quieres modificar alguna columna o configuración? (sí/no)", ["sí", "no"])
        if edit == "no":
            break

        inputs = {f['name']: f for f in config.get("input_features", [])}
        outputs = {f['name']: f for f in config.get("output_features", [])}
        feature_role_options = ["input", "output", "ninguna"]
        input_type_options = [
            "binary", "number", "category", "text", "vector", "image",
            "audio", "timeseries", "date"
        ]
        output_type_options = [
            "binary", "number", "category", "bag", "set", "sequence",
            "text", "vector", "audio", "date", "h3", "image", "timeseries"
        ]

        print("\nColumnas disponibles para modificación:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        sel = input("Selecciona el número de la columna que deseas modificar: ")
        if not sel.isdigit() or int(sel) < 1 or int(sel) > len(df.columns):
            print("Selección inválida.")
            continue
        col_name = df.columns[int(sel) - 1]

        role = interaction.ask_question(f"Uso para la columna '{col_name}':", feature_role_options)

        inputs.pop(col_name, None)
        outputs.pop(col_name, None)

        if role == "input":
            tipo = interaction.ask_question(f"Tipo de dato de '{col_name}':", input_type_options)
            inputs[col_name] = {"name": col_name, "type": tipo}
        elif role == "output":
            tipo = interaction.ask_question(f"Tipo de objetivo de '{col_name}':", output_type_options)
            outputs[col_name] = {"name": col_name, "type": tipo}
        else:
            print(f"Columna '{col_name}' será ignorada.")

        config["input_features"] = list(inputs.values())
        config["output_features"] = list(outputs.values())

    with open("auto_config.yml", "w") as f:
        yaml.dump(config, f, allow_unicode=True)

    print("\nResumen generado por LLaMA:")
    print(wrap_text(generar_resumen(config, config.get("preprocessing", {}))))

if __name__ == "__main__":
    main()
