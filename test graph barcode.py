
# edges are UHDIRECTED, meaning we dont know the direction, but we know the SUBSET.

'''
example 1.
input
node [0,1,2], n nodes
edge [[0,2],[1,2],[2]] the subset for nth node, include self
0--1
1--2
2

expected output [[0,2] [1,2]] (the connection)
0-->2
1-->2
'''


'''
example 2.
input
node [0,1,2], n nodes
edge [[0],[1],[0,1,2]] the subset for nth node.
0
1
2--1
2--0

expected output [[2,0] [2,1]] 
2-->0
2-->1
'''

# basic solution
# 1 step only.

import collections
import numpy as np
from collections import deque


def generate_testset(n, density):
    # n for n nodes. density=1 for fully connected and 0 for no connections.   
    groundtruth=[]
    barcode=collections.defaultdict(set)
    for i in range(n):
        barcode[i].add(i)
        conn=np.random.randint(low=0,high=n,size=np.floor(density*n).astype(int))
        for c in conn:
            # adding connecctions to groud truth
            groundtruth.append([i,c])
            # union for subset
            barcode[i].add(c)

    return n,barcode,groundtruth

def sovler1(n, barcode):
    # solve for 1 next connection problem
    '''
    assume 2 direciton connection, make the graph
    go though the graph again and prune the gragh 
    '''
    # 1st loop, making graph assuming undirected
    graph=collections.defaultdict(set)
    for _, subset in barcode.items():
        subset=list(subset)
        for i in subset:
            for j in subset:
                graph[i].add(j)
                graph[j].add(i)
    # 2nd loopl, pruning
    for start, subset in barcode.items():
        for end in list(graph[start]):
            if end not in subset:
                graph[start].remove(end)
    return graph

# example 
n,barcode,groundtruth=generate_testset(5,0.5)
sovler1(n, barcode)


# branch 1, optimization
'''
opt1
base appoarch is slow, because we do fully connect then prune.
we can sort the subsets, such that more constraints sbuject (smaller size subsets) are applied early.
we record the current set, when a new set comes in, just add the new connections. 
    if interset=curset (newset covers the curset) and union>curset, (newset is larger), we just add the new connection
    if interset<curset, and  union>curset (extending out in a new branch), we add the new connections
and we dont need pruning anymore

but adding new nodes are hard:
        # for a newnode and many old nodes, we dont know who to connect

        # greedy appoarch
        # we save the new nodes to connect, and the corresponding old nodes.
        # next time, do the same thing, but connect the shortest old connections
        
        # backtract
        # incase we want all possible solultion
        # basiclly we try all permutations that satisfy the solution.

'''


def opt1(n, barcode):
    # sort the subsets
    sortedkeys=[]
    for k,v in barcode.items():
        sortedkeys.append((k,len(v)))
    sortedkeys.sort(key=lambda x: x[1])
    sortedkeys=[k[0] for k in sortedkeys]

    graph=collections.defaultdict(set)
    curset=set()
    for k in sortedkeys:
        curset.add(k)
        if curset.union(barcode[k])>=curset and curset.intersection(barcode[k])==curset:
            newnodes=list(curset.union(barcode[k])-curset)
    # not finished

'''
opt2
recursion
we sort the subset and start from the largest subset
we make a recursive function, to convert barcode to graph
    stop condition: when barcode is 1 node left
    childgraph=rec(barcode except cur)
    curgraph.next=childgraph
    return curgraph

'''

class Graph:
    def __init__(self, val=0, next=[]) -> None:
        self.val=val
        self.next=next

def opt2(n, barcode):
    barcode=[list(v) for k,v in barcode.items()]
    # sort the subsets
    sortedkeys=[]
    for k,v in barcode.items():
        sortedkeys.append((k,len(v)))
    sortedkeys.sort(key=lambda x: -x[1])
    sortedkeys=[k[0] for k in sortedkeys]

    def rec(barcode): # have not consider cylic case?
        if len(barcode)==1:
            return Graph(val=barcode[0])
        if len(barcode)==0:
            return None
        headset=set(barcode[0])
        curmatch=set()
        i=1
        while len(headset-curmatch)>1:
            curmatch=curmatch.union(barcode[i])
            i+=1
        head=Graph(val=headset-curmatch)
        childgraph=rec(barcode[1:])
        head.next.append(childgraph)
        return head

    graph=collections.defaultdict(set)
    curset=set()
    for k in sortedkeys:
        curset.add(k)
        if curset.union(barcode[k])>=curset and curset.intersection(barcode[k])==curset:
            newnodes=list(curset.union(barcode[k])-curset)





# branch 2, extend to practise

'''
assume that the edge has strength, and the strengh is in (0,1)
assume the strength is a fix value for all connections, without noise.
then we automatically know the the layers, the barcode data is like level order traverse.
so, with base approach, we fully connect between adj layers, instead of all subsets.
'''

'''
assume that the edge has strength, and the strengh is in (0,1)
assume the strength has noise but it is same across nodes.
then, instead of knowing the layers precisly, we have a probability
we sort the subset to form the level order traversal, then do the same thing.
'''


'''
assume that the edge has strength, and the strengh is in (0,1)
(optional conditions) 
    assume edge strengths are similar for one starting node? 
    assume strength within a finer range like (0.6,0.8)?
    (we dont need this to solve, but i guess thats what we expect in practice)
now, we cont from optimized approach 1, 
    sort the subset and start from the smallest (more constrained) subset
        sort the strength and start from the strongest connection
        for new connections within cursubset, iterate to find a best match such that... (conditions)
    for new subsets, add the new connections to curset
no need for pruning.

complexity:
barcode has n nodes and m subsets
we sort the m subsets, mlogm
we loop all nodes n
for each subset we sort, m n/m log n/m
for each node in each subset, we iterate cur subset or layer, n/m n/m

roughly n2/m2

we create a datastructure for the graph described by subsets
for worset case subsets fully connected? or some other strange case
but roughly within (n,n2)
'''

def generate_testset2(n, density, threshold=0.2):
    # n for n nodes. density=1 for fully connected and 0 for no connections. 
    # now has edge strength, so that we use threshold to group the subsets up to some extend
    groundtruthedge=[]
    groundtruthgraph=collections.defaultdict(set)
    barcode=collections.defaultdict(list)
    for i in range(n):
        barcode[i].append((i,1))
        groundtruthgraph[i].add((i,1))
        conn=np.random.randint(low=0,high=n,size=np.floor(density*n).astype(int))
        for c in conn:
            # adding connecctions to groud truth
            s=np.random.random()
            groundtruthgraph[i].add((c,s))
            groundtruthedge.append([i,c,s]) # input, connected to, strenght
    # bfs for next layer
    q=deque(list(range(n)))
    while q:
        start=q.popleft()
        for (c, s) in groundtruthgraph[start]:
            if s>threshold:
                

    return n,barcode,groundtruth







class UnionFind():    

    def __init__(self,size) -> None:
        self.root=[i for i in range(size)]
        
    def find(self, x):
        # recursive up to find root
        if x == self.root[x]:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x, y):
        # x to y but not y to x
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.root[rootY] = rootX
            
def sovler2(n, barcode):
    '''
    directed union
    '''
    # 1st loop, making graph assuming undirected
    graph=UnionFind(n)
    for start, subset in barcode.items():
        subset=list(subset)
        for i in subset:
            for j in subset:
                graph.union(i,j)
    
    graph.root
    # 2nd loopl, pruning
    for start, subset in barcode.items():
        for end in list(graph[start]):
            if end not in subset:
                graph[start].remove(end)


n,barcode,groundtruth=generate_testset(5,0.5)




'''
6.18

define the problem

question, how does virus works and what is the subset produced by virus?
    seems virus is a tag that let neurons express something we can detect later
    we inject the virus to brain, and initiall it infests several head node
    then headnode express the virus and infest the down streams.

    how does the infest next works?
    hypothesis 1, virus can infest all but has a much higher probablily to infest the next one
    hypothesis 2, virus can only infest next, and the ability/probability to infest next decrease over number of infests. each head may have different inital prob, and each next can have different prob. but we should have a rough estimation of this probabily.
    hypothesis 3, virus can only infest next, and we have a concentration. the virus will need to copy itself to achieve a high enough concentration to infest the next, and we have or do not have ways to detect the concentration. if not, the barcode is binary 01. if yes, the barcode is between a threshold to 1, where the threshold within {0,1}

    current evidence:
        virus only tag several neurons, maybe only 2 steps or 3 step, not infinit
            there is a concentration or probability. if not, the virus should tag much more number of neurons, because we know there are more than 3 neurons connecting
        the detection produce a barcode, currently it is binary. 
            maybe we have a probability. or maybe we have a concentration, but the detection is a threshold. either way we can assume there is a probability
        
     conclusion and summary
        how the virus works
            we have virus that tag serveral heads with different strength(virus concentration)
            the head infest next with a probability of the concentration (high concentration is more likeliy to infest next)
            next then next. till the concentration is low enough (we konw this threshold)
        how do we get the barcode
            we run some detections at nodes and detect the down stream nodes tagged by the virus.
            eg, 1->2->3->4
            if we detect at 1, then we have 1234, the order does not matter
            if we detect at 2, then we have 234
            if at 4, we only have 4.
'''
import collections
import numpy as np
from collections import deque


def generate_testset2(n, density, threshold=0.2):
    # n for n nodes. density=1 for fully connected and 0 for no connections. 
    # now has edge strength, so that we use threshold to group the subsets up to some extend
    groundtruthedge=[]
    groundtruthgraph=collections.defaultdict(set)
    for i in range(n):
        groundtruthgraph[i].add((i,1))
        conn=np.random.randint(low=0,high=n,size=np.floor(density*n).astype(int))
        conn=list(set(conn))
        for c in conn:
            if c !=i:
                # adding connecctions to groud truth
                s=np.random.random()
                groundtruthgraph[i].add((c,s))
                groundtruthedge.append([i,c,s]) # input, connected to, strenght
    # dfs
    barcode=collections.defaultdict(set)
    for i in range(n):
        barcode[i].add(i)
        print('starting head ', i)
        q=[(i,1)]
        while q:
            start,concentration=q.pop()
            print(start,concentration)
            for (c, s) in groundtruthgraph[start]:
                if c!=start and c not in barcode[i]:
                    newconcentration=s*concentration
                    if newconcentration>threshold:
                        barcode[i].add(c)
                        q.append((c, newconcentration))

    return n,barcode,groundtruthedge

generate_testset2(4, 0.5, threshold=0.2)



# 




