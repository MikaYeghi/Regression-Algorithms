close all
clear all
clc

% main code
global data_fraction; % fraction of data that's being used
data_fraction = 1;
file_path = 'california.txt';
D = retrieve_data(file_path); % retrieve data
frac1 = 0.8; % fraction of data that's trainval
frac2 = 0.8; % fraction if trainval that's train
lambda = 1e-3; % lambda for the loss function
[trainval_D, test_D] = random_split(D, frac1);
[train_D, val_D] = random_split(trainval_D, frac2);
w = lp_l1_regression(train_D, lambda) % find solution using linear programming
mae = compute_mean_abs_error(val_D, w)

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

% lp_l1_regression - solves a linear programming minimizations problem
function w = lp_l1_regression(train_D, lambda)
col_num = length(train_D(1,:));
d = col_num - 1; % number of parameters
Phi = train_D(:, 1:d); % matrix of parameters
n = length(Phi); % number of records
y = train_D(:, col_num); % array of final values
A = generate_A(Phi, d, n); % returns the matrix A for linprog
b = generate_b(y, d); % returns the vector b for linprog
f = generate_f(lambda, d, n); % returns the vector f gor linprog

w = linprog(f, A, b); % find w, v and dzeta
w = w(1:d); % extract the values for w

% functions
% generates matrix A for linprog
    function A = generate_A(Phi, d, n)
       A = [eye(d), -eye(d), zeros(d, n)]; % first row
       A = [A; -eye(d), -eye(d), zeros(d, n)]; % second row
       A = [A; Phi, zeros(n, d), -eye(n)]; % third row
       A = [A; -Phi, zeros(n, d), -eye(n)]; % fourth row
    end

% generates vector b for linprog
    function b = generate_b(y, d)
       b = [zeros(d, 1); zeros(d, 1); y; -y]; 
    end

% generates vector f for linprog
    function f = generate_f(lambda, d, n)
    f = [zeros(d, 1); lambda / 2 * ones(d, 1); 1 / n * ones(n, 1)];
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

