function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
%hypothesis=theta(1)+(theta(2)*X(:,2)) + (theta(3)*X(:,3));
hypothesis=X*theta;
%x=X(:,2);
%z=X(:,3);
%a=(x-mean(x))/(max(x)-min(x));
%b=(z-mean(z))/(max(z)-min(z));
%thetazero= theta(1) - (alpha*(1/m)*sum(hypothesis-y));
%thetaone= theta(2) - (alpha*(1/m)*sum((hypothesis-y).*a));
%thetatwo= theta(3) - (alpha*(1/m)*sum((hypothesis-y).*b));
%theta=[thetazero;thetaone;thetatwo];
errors = hypothesis - y;
delta = X' * errors;
theta = theta - (alpha/m) * delta;
  












    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
