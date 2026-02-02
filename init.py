from utils import cmd

REQ_INSTALL = "pip install -r requirements.txt"
PYTORCH_INSTALL = (
    "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
)


def requirements():
    cmd(REQ_INSTALL)
    cmd("clear")
