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

## make status report thing for commitee meeting at feb 15

1. what direction is useful 
2. what groups are the potential targets
3. skill set that i can prepare for them 

current researh, mathmetical mind reading of subject driving in VR
so far more focus on behavior than neural data.

potential directions:

    neural data
    预期只能解释一小部分latent variable， 用处？相同的方法用在eeg？
        1. neural encoding/representation. eg position, uncertainty
        2. neural dynamic, state transition
        3. neural policy


    representation related, relational or casual graph
    广泛来说用处更大？万一能发nips （目前没有graph state with uncertainty RL， 但是拼一个好像不难）
    选择1 拼装几个现有的关于graph的work之后用在这个task
    选择2 搞个全新的representation解决cov size是n**2的问题
    选择3 representation learning with 
        1. representation leraning, efficient representation for multi ff, 
        2. graph representation and dynamic with uncertainty and use in rl
        3. 


    multi sensory integration related
    比较简单 model应该不会动 主要是等新的验证data， 可以是side project
        1. additional experiement to map information source
        2












预算1000-1700的1b+1b+ 想找个性价比高的, 不是想找最便宜的
必要:
大于750 sqft
噪音(这个不好估计是吧, 不是高速路边或者一圈公寓空调外机很吵那种应该都还行)
安全些 车可能被砸那种就算了

加分项:
天花板高或者loft
1楼 出门就是停车位那种
有个阳台 或者1楼有个放椅子的空地
离288近 或者离中国城近

其他备注:
很少去tmc 离的远没关系
停车楼太不方便了 那种可以停门前不远地方的最好了
预算1000-1700, 想找个性价比高的, 不是想找最便宜的



