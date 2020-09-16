current=1200
future=1600

overlap=4

offer=500

def actual_future(future, offer):
    return (future*12-offer)/12

def days(future, current,overlap):
    cost=current*overlap/30
    inviscost=future*overlap/30
    return (cost+inviscost)/12