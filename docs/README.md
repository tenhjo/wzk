**WZK**
---
![WerkZeugKasten Logo](wzk.png)

Abbreviation of the German noun *"WerkZeugKasten"* (~*"toolbox"*)

---
A mixed collection of "fairly general" convenience functions I use across different projects.
The functions are structured with respect to the existing module they extend, or the functionality they facilitate. 

# Attention
This module has wrappers for multiprocessing and matplotlib
and sets some backend flags for those. 
This should not interfere with anything else but be aware of it.
* matplotlib
  * sets backend via mpl.use('TkAgg')
* multiprocessing
  * limit the threads used by numpy / scipy to 1 [see StackOverflow](https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading)
  ```
  os.environ['MKL_NUM_THREADS'] = '1'
  os.environ['OPENBLAS_NUM_THREADS'] = '1'
  os.environ['NUMEXPR_NUM_THREADS'] = '1'
  os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
  os.environ['OMP_NUM_THREADS'] = '1'
  ```

make sure you import wzk before matplotlib / numpy to get the expected behavior

---
This repo has nothing to do with [Werkzeug](https://pypi.org/project/Werkzeug/). 
They just seem to have the same humor.
