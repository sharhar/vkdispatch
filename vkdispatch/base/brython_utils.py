import sys

def is_brython() -> bool:
    return sys.implementation.name == "Brython"