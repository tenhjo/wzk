import os
# Numpy uses multiprocessing to speed up matrix calculations
# https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
# https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
