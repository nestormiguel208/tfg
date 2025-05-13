from utils.rag_utils import ayuda_mode

def ask_question(prompt, options=None, vectordb=None):
    while True:
        print(f"\n {prompt}")
        if options:
            for i, option in enumerate(options, 1):
                print(f"  {i}. {option}")
            user_input = input("Selecciona una opción (número o 'ayuda: pregunta'): ")
            if user_input.lower().startswith("ayuda:") and vectordb:
                ayuda_mode(user_input.replace("ayuda:", "").strip(), vectordb)
                continue
            if user_input.isdigit():
                idx = int(user_input)
                if 1 <= idx <= len(options):
                    return options[idx - 1].split(':')[0].strip()
            print(" Entrada inválida.")
        else:
            user_input = input("Tú: ")
            if user_input.lower().startswith("ayuda:") and vectordb:
                ayuda_mode(user_input.replace("ayuda:", "").strip(), vectordb)
                continue
            return user_input
