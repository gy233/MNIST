function cost=QuadraticCost_fn(a,y)
% Return the cost
% a: output
% y: desired output
cost=0.5*sum((a-y).^2);