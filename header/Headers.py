import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import logging
import inspect

LEARNING_RATE_THETA0 = 0.001
LEARNING_RATE_THETA1 = 0.0000000000001
CONVERGENCE = 0.00001
EXP_COL = ['km', 'price']
