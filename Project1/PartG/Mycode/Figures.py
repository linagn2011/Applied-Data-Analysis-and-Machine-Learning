# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 19:10:47 2023

@author: eirik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib import cm
import os

# Set path to save the figures and data files
FIGURE_PATH = "./Figures"


def fig_path(fig_id):
    """

    """
    return os.path.join(FIGURE_PATH + "/", fig_id)

