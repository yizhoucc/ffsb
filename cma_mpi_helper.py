import multiprocessing
import multiprocess as mp
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial


# usages:
# run_pool(eval function)
# returns result

"""
exmaple:
from cma_mpi_helper import run
import torch
def f(x):
    return sum(x)
    
if __name__=="__main__":
    input=torch.rand(10,2).tolist()
    res=run(f,input)
    print(res)
"""

def worker(x):
    return x

def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(4)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        print("Aborting due to timeout")
        raise

def run_pool_(worker, callback=None):
    if not callback:
        def callback(item,log):
            log.append(item)
    pool = mp.Pool(processes=4,maxtasksperchild=1)
    results = []
    for i in range(5):
        tmpcallback=lambda x: callback(x,results)
        abortable_func = partial(abortable_worker, worker, timeout=3)
        pool.apply_async(abortable_func, i, callback=tmpcallback)
    pool.close()
    pool.join()
    return results

def run_pool(worker,workerparam,processes=4):
    pool = mp.Pool(processes=processes,maxtasksperchild=1)
    results = []
    for param in workerparam:
        abortable_func = partial(abortable_worker, worker, timeout=10)
        pool.apply_async(abortable_func, param)
    pool.close()
    pool.join()
    return results

def run(f,params,numprocess=4):
    with mp.Pool(processes=numprocess) as pool:
        res=pool.map(f,params, 1)
        pool.close()
        pool.join()
    return res