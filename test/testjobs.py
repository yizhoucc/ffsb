import datajoint as dj
from datajoint.jobs import key_hash
import os
import time
import multiprocessing as mp

dj.config['database.user']="yizhou"
dj.config['database.host']="at-database.ad.bcm.edu"
dj.config['database.password']="yizhou#1"
schema=dj.schema("yizhou_test",locals(),create_tables=True)


@schema
class X(dj.Manual): 
    definition = """ 
        id: int auto_increment # job index
        ---
        x: float
        iter: int
        """

@schema
class Y(dj.Computed): 
    definition = """ 
        -> X
        ---
        x: float
        y: float
        # seems dj handels the reservation alrady
        # done: tinyint # use 0 and 1 for bool
        # doing: tinyint # use 0 and 1 for bool
        iter: int
        """
    def make(self, key):
        def _somecostlyfunction(x):
            time.sleep(1)
            return 2*x
        x, iteration =(X() & key).fetch1('x','iter')
        y=_somecostlyfunction(x)
        res = key.copy()
        res['y']=y
        self.insert1(res)



        

