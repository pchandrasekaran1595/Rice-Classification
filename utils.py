import os
from termcolor import colored
os.system("color")

#####################################################################################################

def myprint(text: str, color: str) -> None:
    print(colored(text=text, color=color))


def breaker(num=50, char="*") -> None:
    myprint("\n" + num*char + "\n", "magenta")


def debug(text: str):
    myprint(text, "red")

#####################################################################################################

SEED = 0
DATA_PATH = "./Data"
MODEL_PATH = "./Models"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

#####################################################################################################
