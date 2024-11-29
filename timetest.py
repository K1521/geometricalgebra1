import time


def max_seconds(max_seconds, interval=1):
    current = start_time = time.time()
    end_time = start_time + max_seconds
    while  current<=end_time:
        yield current-start_time
        now=time.time()
        print(end_time-now)
        time.sleep(max(0,min(end_time-now,interval+current-now)))
        current=time.time()
    yield current-start_time

for i in max_seconds(10):
    time.sleep(0.7)
    print(i)