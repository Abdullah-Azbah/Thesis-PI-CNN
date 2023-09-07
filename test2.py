import numpy as np
import tensorflow.experimental.numpy as tnp

data = np.array([
    [1.0, 2.0, 4.0],
    [7.0, 11.0, 16.0]
])

print(np.gradient(data, axis=1))
print(tnp.gradient(data, axis=1))