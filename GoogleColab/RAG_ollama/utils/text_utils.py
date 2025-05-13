import textwrap

def wrap_text(text, width=200):
    return "\n".join(textwrap.wrap(text, width))
