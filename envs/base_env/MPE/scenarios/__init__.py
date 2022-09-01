import imp
import os.path as osp

def load(name):
    name = "simple_spread.py"
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)



