function y=softmax(x)
y=exp(x)./sum(exp(x)); % to prevent data overflow from happening, we use x-max(x)