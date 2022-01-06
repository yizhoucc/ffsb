# todo list





## how to use mk stop in logll calculation




## some plot functions described in paper git




## skip analysis
assuming that the trial type is dependent, such that type(1)->type(2)->type(3), we can solve for a hidden state (value) and predict the next trial type given the previous types.
an rnn approach will be, n to one prediction, and n in range of 0 to len(data)

baysien approach is more straight forward:
if we are given 1:n-1 trials and we want to predict n trial type, we have p(n)=p(1).p(2|1).p(3|1,2)...p(n|1:n-1)
and we approximate the 1:n-1 prob as f(1:n-1) (this is more appropate since n is large)

if we want best explainability, thats baysien and liner regresssion.
we define type1 to type2 prob as p12, and so on for all other types.
we then define an n, meaning how many trial we are concidering.
if n=1, we can use the probs directly to predict the next trial type.
if n=2, we have type(2)=p(type(2)|type(1),type(0))*p(type(1)|type(0)), since we already given type(1) and type(0) to predict type(2), type(2)=p(type(2)|type(0),type(1))
with linear regression, we have f(type(0),type(1))=w0.type(0) + w1.type(1) +b
find the w and b, we predict the type(2)
if n=3,4,5..., the logic is similar.

reformat the question:
we have a sequance of categorical variables.
we think this sequence is markovian, previous affect current and future.
we hope to have an explainable model to describe how previous seq affects post seq (or point).
eg, rnn seq to one, hard to explain what hidden state means?
eg, approximate p(n|1:n-1) as f(1:n-1)=sum(w_1:n*1:n + b) to predict p(n), is this still markovian?

example:
one hot encoding
t0=[0010]
t1=[1000]
t2=[0010]
t3=[1000]
....
t=?

## 