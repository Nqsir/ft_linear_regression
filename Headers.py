import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import logging
LEARNING_RATE_THETA0 = 0.001
LEARNING_RATE_THETA1 = 0.0000000000001
CONVERGENCE = 0.01
EXP_COL = ['km', 'price']