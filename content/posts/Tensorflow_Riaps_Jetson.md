---
title: "Tensorflow not initializing during riaps deployment"
date: 2019-09-08T17:23:18+03:00
draft: false
---

# Strange error cannot allocate memory in static TLS block under riaps plateform

I am deploying on an nvidia Jetson Xavier a Tensorflow / Keras application _(tensorflow-gpu==1.13.1+nv19.5)_. 

If I launch my python code outside riaps, it's working , no error. But when I deploy it with riaps_ctrl and launch it, here is the result :




```

<class 'ImportError'>: Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 28, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
  File "/usr/lib/python3.6/imp.py", line 243, in load_module
    return load_dynamic(name, filename, file)
  File "/usr/lib/python3.6/imp.py", line 343, in load_dynamic
    return _load(spec)
ImportError: /usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block


Failed to load the native TensorFlow runtime.

See https://www.tensorflow.org/install/errors

for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.
Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 28, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
  File "/usr/lib/python3.6/imp.py", line 243, in load_module
    return load_dynamic(name, filename, file)
  File "/usr/lib/python3.6/imp.py", line 343, in load_dynamic
    return _load(spec)
ImportError: /usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block

```

Impossible to update the jetson tensorflow version, because cudnn installed version is 7.3.1 and I don't want to re-flash the device and lose all my riaps installation. If I use a more recent version of tensorflow I have this error:

```
Loaded runtime CuDNN library: 7.3.1 but source was compiled with: 7.5.0

```


The solution I found is to move the Tensorflow include at the very top of the python code :

```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Input, Dropout, Conv2D, BatchNormalization, MaxPool2D, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import MaxPooling1D, Conv1D, GlobalAveragePooling1D, Reshape


from riaps.run.comp import Component
import logging
from random import random
import time as t

from time import mktime
from datetime import datetime
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import normalize

```

If the import tensorflow are after the sklearn import, it's not working.
This page gave me the hint : https://www.mail-archive.com/python-devel@lists.fedoraproject.org/msg01516.html

I think this bug is specific to the version of tensorflow I have on the jetson because the raspberry pi doesn't have this issue.