import time


def st_time(func):
    """
        st decorator to calculate the total time of a func
    """

    def st_func(*args, **keyArgs):
        t1 = time.perf_counter()
        r = func(*args, **keyArgs)
        t2 = time.perf_counter()
        print("Function=%s, Time=%s" % (func.__name__, t2 - t1))
        return r

    return st_func
