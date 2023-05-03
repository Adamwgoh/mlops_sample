import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from common.general_util import get_logger

def main():
    print("Hello world")

if __name__=="__main__":
    main()