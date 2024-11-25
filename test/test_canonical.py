import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)
from src import UMPS


if __name__ == '__main__':
    D = 30
    d = 2
    mps = UMPS.random(D, d)
    mps.leftCanonical()
