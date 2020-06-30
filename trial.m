clear all; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha).
%
%               Your task is to first make sure that your functions -
%               computeCost and gradientDescent already work with
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 1; % modified from 0.01 because 3.2.1
num_iters = 50; %modified from 100 because 3.2.1

% Init Theta and Run Gradient Descent
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% selecting learning rates.
% 1 is best.
% hold on;
% theta = zeros(3, 1);
% [theta,  J_history] = gradientDescentMulti(X, y,  theta,  alpha/10.0,  num_iters);
% plot(1:numel(J_history), J_history, '-y', 'LineWidth', 2);
% fprintf('Theta computed from gradient descent: \n');
% fprintf(' %f \n', theta);
% fprintf('\n');


% theta = zeros(3, 1);
% [theta,  J_history] = gradientDescentMulti(X, y,  theta,  alpha/100.0,  num_iters);
% plot(1:numel(J_history), J_history, '-r', 'LineWidth', 2);
% fprintf('Theta computed from gradient descent: \n');
% fprintf(' %f \n', theta);
% fprintf('\n');


% theta = zeros(3, 1);
% [theta,  J_history] = gradientDescentMulti(X, y,  theta,  alpha/1000.0,  num_iters);
% plot(1:numel(J_history), J_history, '-k', 'LineWidth', 2);
% fprintf('Theta computed from gradient descent: \n');
% fprintf(' %f \n', theta);
% fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = [1 (1650-mu(1))/sigma(1) (3-mu(2))/sigma(2)]*theta; % You should change this
% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);
fprintf('Program paused. Press enter to continue.\n');
pause;

% Plotting Training and regressioned data.
fprintf('Plotting Training and regressioned results by gradient descent.\n');
X = [ones(m, 1) data(:, 1:2)]; %denormalize features
figure;
plot3(X(:,2),X(:,3),y,"o");
xlabel('sq-ft of room');
ylabel('#bedroom');
zlabel('price');
grid;
hold on;
xx = linspace(0,5000,25);
yy = linspace(1,5,25);
zz = zeros(size(xx,2),size(yy,2));
for i=1:size(xx,2)
for j=1:size(yy,2)
  zz(i,j) = [1 (xx(i)-mu(1))/sigma(1) (yy(j)-mu(2))/sigma(2)]*theta;
end
end
mesh(xx,yy,zz);
title('Result of Gradient Descent');
legend('Training data', 'Linear regression');

fprintf('Program paused. Press enter to continue.\n');
pause;