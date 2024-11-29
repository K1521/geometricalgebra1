
import numpy as np
import time


large_array = np.random.rand(10**6)
# Measure time for in-place addition with += 0
start_time_0_inplace = time.time()
for i in range(1000):
    large_array += 0
time_0_inplace = time.time() - start_time_0_inplace

# Reset the large array for a fair comparison
large_array = np.random.rand(10**6)

# Measure time for in-place addition with += 1
start_time_1_inplace = time.time()
for i in range(1000):
    large_array += 1
time_1_inplace = time.time() - start_time_1_inplace

print(time_0_inplace, time_1_inplace)


import numpy as np
import time

def fastsum(x):
    x = iter(x)
    try:
        s = next(x)
    except StopIteration:
        return 0
    
    try:
        s += next(x)
    except StopIteration:
        return s
    
    for t in x:
        s += t
    
    return s

# Generate test data
sizes = [10000]  # Different sizes for testing
results = {}

for size in sizes:
    data = [np.random.rand(size) for i in range(10)]

    
    # Measure time for the built-in sum function
    start = time.time()

    for i in range(10000):sum_result = np.sum(data)
    end = time.time()
    built_in_time = end - start
    print("np",built_in_time)
    
    # Measure time for fastsum function
    start = time.time()
    for i in range(10000):fastsum_result = fastsum(data)
    end = time.time()
    fastsum_time = end - start
    print("fastsum_time",fastsum_time)

    start = time.time()
    for i in range(10000):fastsum_result = sum(data)
    end = time.time()
    sum_time = end - start
    print("sum_time",fastsum_time)
    
    



