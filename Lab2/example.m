%% HW2 - CIFAR-10 NN inference (example)
% Top function must be:  function prob = M11XXXXXX(img, net)
%   img  : 32x32x3 double, pixel value in [0,1]
%   net  : network loaded from HW2_model.mat
%   prob : 10x1 probability (NOT the category)
% Do NOT use any NN library function. Do NOT change the Data load part.
clear; clc;

%% ======================= Data load (do not change) =======================
S   = load('HW2_model.mat');
net = S.convnet;                          % use net.Layers(k).Weights / Bias ...

D      = load('test_image.mat');          % data: 1000x3072 uint8, labels: 1000x1 uint8
numImg = size(D.data, 1);
labels = double(D.labels);                % class 0 ~ 9
labelText = ["airplane","automobile","bird","cat","deer", ...
             "dog","frog","horse","ship","truck"];

imgs = zeros(32, 32, 3, numImg);
for n = 1:numImg
    % CIFAR row (R,G,B planes, row-major) -> 32x32x3 image, scaled to [0,1]
    imgs(:,:,:,n) = double(permute(reshape(D.data(n,:), 32, 32, 3), [2 1 3])) / 255;
end
% example: imshow(imgs(:,:,:,1)); title(labelText(labels(1)+1));
%% ==========================================================================

%% ======================= Evaluation (do not change) =======================
pred = zeros(numImg, 1);
tic;
for n = 1:numImg
    prob = M11XXXXXX(imgs(:,:,:,n), net);
    if ~isequal(size(prob), [10 1])
        error('prob must be 10x1, got %dx%d.', size(prob,1), size(prob,2));
    end
    if abs(sum(prob) - 1) > 1e-4 || any(prob < 0)
        warning('Image %d: prob is not a valid probability (sum = %.4f).', n, sum(prob));
    end
    [~, k] = max(prob);
    pred(n) = k - 1;                      % class 0 ~ 9
end
t = toc;

C = zeros(10);                            % row = true, col = predicted
for n = 1:numImg
    C(labels(n)+1, pred(n)+1) = C(labels(n)+1, pred(n)+1) + 1;
end

fprintf('\n #  Class        Images  Accuracy\n');
fprintf('--  ----------  ------  --------\n');
for k = 1:10
    fprintf('%2d  %-10s  %6d  %7.2f%%\n', k-1, labelText(k), ...
            sum(C(k,:)), 100*C(k,k)/max(sum(C(k,:)), 1));
end
fprintf('--  ----------  ------  --------\n');
fprintf('    %-10s  %6d  %7.2f%%\n', 'Total', numImg, mean(pred == labels)*100);
fprintf('\nTime : %.2f s\n', t);

%% ======================= Your code starts here ============================
% Only implement M11XXXXXX below (you may add your own local functions).
% Do NOT use any NN library function.

function prob = M11XXXXXX(img, net)
    % img  : 32x32x3 double in [0,1]
    % net  : network from HW2_model.mat
    % prob : 10x1 probability
    % TODO: implement the forward propagation by yourself
    prob = ones(10, 1) / 10;
end
