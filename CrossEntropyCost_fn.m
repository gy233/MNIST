function cost=CrossEntropyCost_fn(a,y)
% Return the cost
% a: output
% y: desired output
cost=-sum(y.*log(a)+(1-y).*log(1-a));