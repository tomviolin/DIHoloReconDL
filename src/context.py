# -*- coding: utf-8 -*-
"""
Adds folder containing source of PyHoloscope to path.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import sys, os

exampledir = os.path.dirname(__file__)
os.chdir(exampledir+"/..")
srcdir = "src"
sys.path.insert(0, os.path.abspath("PyHoloscope/src"))
