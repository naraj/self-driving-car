from sys import argv
import IPython.nbformat.current as nbf
nb = nbf.read(open(argv[1], 'r'), 'py')
nbf.write(nb, open(argv[2], 'w'), 'ipynb')
