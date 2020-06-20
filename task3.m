close all
clear all
clc

% main code
clear
file_path = 'california.txt'; % path to the file
global data_fraction; % fraction of data that's being read
data_fraction = 1;
D = retrieve_data(file_path); % retrieve entire data into D
frac1 = 0.8;
frac2 = 0.8;
global etha
global tau
global grad_cutoff
global max_iter
etha = 1e-2; % optimization hyperparameter
tau = 1e-5; % tau for smoothing
grad_cutoff = 5e-6; % gradient below this value is considered to be zero
max_iter = 2e3; % maximum number of iterations in gradient descent

%lambdas = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3];
lambda = 1e-3;
[trainval_D, test_D] = random_split(D, frac1); % split data set into trainval/test sets
[train_D, val_D] = random_split(trainval_D, frac2); % split trainval set into train/val sets
w = smoothed_l1_regression(train_D, lambda)
mae = compute_mean_abs_error(val_D, w)
%best_lambda = find_best_lambda(train_D, val_D, lambdas)
% w = smoothed_l1_regression(train_D, best_lambda)
% mae = compute_mean_abs_error(val_D, w)


% functions
% retrieve_data - retrieves data from .txt file
function data = retrieve_data(file_path)
global data_fraction;
fileID = fopen(file_path, 'r'); % open the file
file_line = fgetl(fileID); % get the 1st line
data = str2num(file_line); % write the 1st line
while ischar(file_line) % loop while the line read from the file is a list of characters (i.e. contains smth)
    file_line = fgetl(fileID); % get new line
    if file_line ~= -1 % if the line contains numbers
        new_row = str2num(file_line); % turn the string row into an array of doubles
        data = [data; new_row]; % append the new record to the existing matrix
    end
end
fclose(fileID); % close the file
data = data(1:(round(length(data) * data_fraction)), :);
%need to normalize the data (parameters, not the output value)
i = 1; % counter
while i <= length(data(1,:))  % go through all columns except for the last one
    column = data(:, i); % select a column
    mean = sum(column) / length(column); % find the mean
    column = column - mean; % subtract mean
    sigma = 0; % standard deviation
    j = 1; % counter
    while j <= length(column)
        sigma = sigma + column(j) ^ 2;
        j = j + 1;
    end
    sigma = sqrt(sigma / length(column)); % compute the standard deviation
    column = column / sigma; % divide by standard deviation
    data(:, i) = column; % replace the original parameter
    
    i = i + 1;
end
end

% random_split - splits entire data into training and test datasets
function [train_set, test_set] = random_split(data_set, frac)
% variables
data_set_n = length(data_set); % length of the data set
train_D_n = round(data_set_n * frac); % length of the training data set
train_set = []; % training data
test_set = []; % testing data

% main code
train_set_indices = randperm(data_set_n, train_D_n); % generate a random set of indices
data_set_indices = 1:length(data_set); % all indices

i = 1; % counter
while i <= length(train_set_indices) % negating train_set indices
    data_set_indices(train_set_indices(i)) = -1 * data_set_indices(train_set_indices(i)); % negate all indices that describe test data
    i = i + 1;
end

i = 1; % counter
while i <= length(data_set_indices)
    if abs(data_set_indices(i)) == data_set_indices(i) % if it's a positive index, then append to test_set
        test_set = [test_set; data_set(data_set_indices(i),:)]; % appending to test_set
    else % if it's a negative index, then append to train_set
        train_set = [train_set; data_set(abs(data_set_indices(i)),:)]; % appending to train_set
    end
    i = i + 1;
end

end

% smoothed_l1_regression - find best estimate based on l1 regression
function w = smoothed_l1_regression(train_set, lambda)
% variables
col_num = length(train_set(1, :)); % number of columns
global etha
global grad_cutoff
global max_iter
Phi = train_set(:, 1:(col_num - 1)); % forming matrix Phi
y = train_set(:, col_num); % forming vector y
n = length(Phi); % number of records
d = length(Phi(1, :)); % number of parameters
w = zeros(d, 1); % initialize zero vector for w

grad = compute_gradient(w, lambda, Phi, y, n, d); % compute initial gradient
cost = compute_objective_function(w, lambda, Phi, y, n); % compute initial objective function
change = 1e6; % variables standing for the change of the cost function over one iteration. Initially set to a very large value
figure;
grid on;
str = sprintf('lambda = %d', lambda);
title(str);
xlim([0 max_iter]);
ylim([0 cost]);
xlabel('Iterations')
ylabel('Objective function')
i = 1; % counter
while i <= max_iter && abs(change) >= grad_cutoff
    grad = compute_gradient(w, lambda, Phi, y, n, d); % update gradient
    change = cost;
    old_w = w; % old_w store the previous value of w
    w = w - etha * grad; % update vector w
    old_cost = cost; % old_cost stores the previous value of cost
    cost = compute_objective_function(w, lambda, Phi, y, n); % update current cost function
    change = change - cost;
    
    % outputting the change value
    clc;
    change
    
    % plotting
    hold on
    plot([i - 1, i], [old_cost, cost], 'r');
    drawnow;
    
    i = i + 1; % counter increment
end

% compute_gradient - computes gradient for given w
    function grad = compute_gradient(w, lambda, Phi, y, n, d)
        grad = zeros(d, 1);
        j = 1;
        while j <= n
            grad = grad + sign(Phi(j, :) * w - y(j)) * Phi(j, :)'; % second term
            j = j + 1;
        end
        grad = grad / n;
        grad = grad + lambda / 2 * sign(w); % first element
    end

% compute_objective_function - computes objective function for given w
    function J = compute_objective_function(w, lambda, Phi, y, n)
        J = 0; % initialize at 0
        
        % 2nd term
        j = 1;
        while j <= n
            J = J + abs(Phi(j, :) * w - y(j));
            j = j + 1;
        end
        J = J / n;
        
        % 1st term
        J = J + lambda / 2 * (sign(w)' * w);
    end
end

% compute_mean_abs_error - computes mean absolute error
function mae = compute_mean_abs_error(val_D, w)
col_num = length(val_D(1,:));
A = val_D(:, 1:(col_num - 1)); % form matrix A
b = val_D(:, col_num); % form vector b
n = length(A); % number of records

estimate_b = A * w; % estimated values
mae = sum(abs(estimate_b - b)); % difference between estimates and real values
mae = mae / n;
end

% find_best_lambda - returns lambda, for which the data set has
% statistically least error
function best_lambda = find_best_lambda(train_set, val_set, lambdas)
num_of_lambdas = length(lambdas); % number of lambdas to be tested
errors = zeros(1, num_of_lambdas); % array of average errors for each lambda. Parallel to lambdas

i = 1; % counter
while i <= num_of_lambdas
    lambda = lambdas(i); % pick a value for lambda from lambdas
    w = smoothed_l1_regression(train_set, lambda); % find the best estimate
    error = compute_mean_abs_error(val_set, w); % compute error for this estimate
    errors(i) = error; % write down to error
    i = i + 1;
end

errors;
[~, best_lambda_index] = min(errors);
best_lambda = lambdas(best_lambda_index);

end