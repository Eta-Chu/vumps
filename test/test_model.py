import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
print(project_root)
from vumps import UniformMPS, SpinHalfXXZChain


if __name__ == '__main__':
    D = 30
    d = 2
    mps = UniformMPS.random(D, d)
    mps.mixCanonical()
    model = SpinHalfXXZChain(delta=1)
    e = model.energy(mps)
    print(e)