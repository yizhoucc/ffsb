# import multiprocessing
# import multiprocessing as mp
# from multiprocessing.dummy import Pool as ThreadPool
# from functools import partial

# def worker(x):
#     return x

# def abortable_worker(func, *args, **kwargs):
#     timeout = kwargs.get('timeout', None)
#     p = ThreadPool(4)
#     res = p.apply_async(func, args=args)
#     try:
#         out = res.get(timeout)  # Wait timeout seconds for func to complete.
#         return out
#     except multiprocessing.TimeoutError:
#         print("Aborting due to timeout")
#         raise



# def pool_test(worker):

#     def callback(item,log):
#         log.append(item)
#     pool = mp.Pool(processes=4,maxtasksperchild=1)
#     results = []
#     tmpcallback=lambda x: callback(x,results)
#     abortable_func = partial(abortable_worker, worker, timeout=3)
#     for i in range(5):
#         pool.apply_async(abortable_func, (i,2*i), callback=tmpcallback)
#     pool.close()
#     pool.join()
#     print(results)


# if __name__ == '__main__':
#     pool_test(worker)

from cma_mpi_helper import run
import torch
def f(x):
    return sum(x)
    
if __name__=="__main__":
    input=torch.rand(10,2).tolist()
    res=run(f,input)
    print(res)