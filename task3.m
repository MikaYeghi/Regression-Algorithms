close all
clear all
clc

% main code
clear
file_path = 'california.txt'; % path to the file
global data_fraction; % fraction of data that's being read
data_fraction = 0.01;
D = retrieve_data(file_path); % retrieve entire data into D
frac1 = 0.8;
frac2 = 0.8;
global etha
global tau
global grad_cutoff
global max_iter
etha = 1e-5; % optimization hyperparameter
tau = 1e-5; % tau for smoothing
grad_cutoff = 1e-6; % gradient below this value is considered to be zero
max_iter = 1e3; % maximum number of iterations in gradient descent
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
col_num = length(train_set); % number of columns
global etha
global tau
global grad_cutoff
global max_iter
A = train_set(:, 1:(col_num - 1)); % forming matrix A
b = train_set(:, (col_num - 1):col_num); % forming vector b
















% variables
col_num = length(train_set(1,:)); % number of columns

A = train_set(:, 1:(col_num - 1)); % form matrix A
b = train_set(:, col_num); % form vector b
w_0 = zeros(col_num - 1, 1); % initial w_0
w_old = w_0; % first w_old
f_ith = (compute_objective_function(A, b, w_0, lambda)); % this matrix will store f_ith's
change = 1e4; % set to some random large value here

close all;
figure;
str = sprintf('lambda = %d', lambda);
title(str);
xlim([0 max_iter]);
%ylim([0 f_ith(1)]);
xlabel('Iterations')
ylabel('Objective function')
i = 2; % counter
while i <= max_iter && change >= grad_cutoff % allow at most 500 iterations
    grad = compute_gradient(A, b, w_old, tau);
    w_new = w_old - etha * grad; % find the new value of w
    w_old = w_new; % update the value of w_old
    
    f_ith = [f_ith, compute_objective_function(A, b, w_new, lambda)]; % update array f_ith
    change = f_ith(i - 1) - f_ith(i);
    % plotting
    hold on
    plot([i - 1, i], [f_ith(i - 1), f_ith(i)], 'r'); % plotting
    drawnow;
    
    i = i + 1;
end

w = w_new;

% nested functions
    function f = compute_objective_function(A, b, w, lambda)
        f = 0;
        n = length(A);
        q = 1; % counter
        while q <= n
            f = f + abs(A(q,:) * w - b(q));
            q = q + 1;
        end
        f = f / n * 2 / lambda; % times 2/lambda since later it's multiplied by lambda/2
        q = 1;
        while q <= length(w)
            f = f + abs(w(q));
            q = q + 1;
        end
        f = f * lambda / 2;
    end

    function grad_f = compute_gradient(A, b, w, tau)
        n = length(A); % number of records
        d = length(w); % number of parameters
        grad_f = zeros(d, 1); % initialize the gradient vector
        first_component = lambda / 2 * sign(w); % lambda component; 1 if w(i) > 0, -1 if <0, 0 if 0
        second_component = grad_f; % n component
        
        k = 1; % counter
        while k <= n
            power = (A(k,:) * w - b(k)) / tau;
            second_component = second_component + tanh(power) * A(1,:)';
            k = k + 1;
        end
        second_component = second_component / n;
        
        grad_f = first_component + second_component;
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