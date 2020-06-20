close all
clear all
clc

% main code
clearvars
global etha_0
global etha_th
global etha_decr
global tau
global max_iter
file_path = 'california.txt';
global data_fraction; % fraction of data that's being read
data_fraction = 0.1;
D = retrieve_data(file_path);
frac1 = 0.8;
frac2 = 0.8;
etha_0 = 1e-1; % initial learning rate
etha_th = 1e-10; % threshold value for etha
etha_decr = 5e-5; % decrement for etha at each iteration
tau = 1e-5;
max_iter = 3e3; % maximum number of iterations

lambda = 1e-3;
[trainval_D, test_D] = random_split(D, frac1);
[train_D, val_D] = random_split(trainval_D, frac2);
w = stochastic_smoothed_l1_regression(train_D, lambda)

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

% stochastic_smoother_l1_regression - finds the best approximation based on
% stochastic gradient descent
function w = stochastic_smoothed_l1_regression(train_set, lambda)
global etha_0
global etha_th
global tau
global max_iter
col_num = length(train_set(1,:));  % number of columns
A = train_set(:, 1:(col_num - 1)); % form matrix A
b = train_set(:, col_num); % form vector b
d = col_num - 1; % number of parameters
n = length(A); % total number of records
w_0 = zeros(d, 1);
w_old = w_0;
k = 0;
while k == 0
    k = round(rand * n); % generate a random index
end
f_old = compute_mean_abs_error(train_set, w_old);
f_new = f_old;
adaptive_rate_data = zeros(n, 1);  % each weight has its own learning rate
etha = etha_0;

% plotting setup
close all;
figure;
str = sprintf('lambda = %d', lambda);
title(str);
xlim([0 max_iter]);
%ylim([0 f_old]);
xlabel('Iterations')
ylabel('Error')

% looping
i = 2; % counter
while i <= max_iter && etha >= etha_th
    k = round(rand * n); % generate a random index
    while k == 0
        k = round(rand * n);
    end
    
    grad = compute_gradient(w_old, lambda, A(k,:)', b(k), tau); % compute the gradient
    
    adaptive_rate_data(k) = adaptive_rate_data(k) + grad'*grad;
    %etha = etha_0 / (sqrt(adaptive_rate_data(k)) + 1e-6);
    etha = etha_0 / i;
    
    w_new = w_old - etha * grad; % compute new w
    w_old = w_new; % update the value of w_old
    f_new = compute_mean_abs_error(train_set, w_new);
    
    % plotting
    hold on
    plot([i - 1, i], [f_old, f_new], 'r');
    drawnow;
    
    f_old = f_new; % update the value of f_old only after plotting
    
    
    i = i + 1;
end

w = w_new;

% nested functions
    function f = compute_objective_function(w, lambda, Phi_x, b_i)
        f = lambda / 2 * sum(abs(w)) + abs(w' * Phi_x - b_i);
    end

    function grad_f = compute_gradient(w, lambda, Phi_x, b_i, tau)
        grad_f = lambda / 2 * sign(w) + tanh((w' * Phi_x - b_i) / tau) * Phi_x;
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