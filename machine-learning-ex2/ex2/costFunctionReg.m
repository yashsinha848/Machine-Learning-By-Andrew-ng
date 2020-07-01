function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

shift_theta = theta(2:size(theta));
theta2 = [0;shift_theta];
h=1./(1+e.^(-X*theta));
a=y.*log(h);
b =(1-y).*log(1-h);
J=(-1/m)*sum(a+b)  +(lambda/(2*m))*sum(theta2.^2);
%J(1)=(-1/m)*sum(a+b);
grad=(1/m)*(X'*(h-y)) +(lambda/(m))*theta2 ;


%grad(1)=(1/m)*sum((h-y).*X);





% =============================================================

end
