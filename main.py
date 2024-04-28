import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes

if __name__ == '__main__':
    x = np.arange(30, dtype=np.float16).reshape(3, 5, 2)
    print(x.dtype)