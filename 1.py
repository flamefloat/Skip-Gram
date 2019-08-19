import numpy as np
random_index = np.random.choice(range(20), 5, replace=False)
print('random_index:', random_index)
a = np.eye(5)[0]
b = np.eye(5)[3]
print(a,b)