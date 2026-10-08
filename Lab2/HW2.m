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
    prob = M11507403(imgs(:,:,:,n), net);
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

function prob = M11507403(img, net)
    % 輸入：32×32×3 圖片，數值已經在 [0,1]
    % 輸出：10×1 類別機率

    %% 1. 輸入層：減去模型使用的平均值
    x = double(img);
    x = x - double(net.Layers(1).Mean);

    %% 2. 三組：卷積 → ReLU → 最大池化
    % 卷積層位於第 2、5、8 層
    for layerIndex = [2, 5, 8]
        x = myConv2D(x, net.Layers(layerIndex));

        % 自行實作 ReLU：負數變成 0
        x = max(x, 0);

        % 對應的池化層位於第 4、7、10 層
        x = myMaxPool(x, net.Layers(layerIndex + 2));
    end

    %% 3. 第一個全連接層：4096 → 512
    x = x(:);

    W1 = double(net.Layers(11).Weights);
    b1 = double(net.Layers(11).Bias);

    x = W1 * x + b1(:);

    %% 4. 第 12 層是 Dropout
    % 現在進行推論，直接保留 x，不隨機刪除數值。

    %% 5. 第二個全連接層：512 → 10
    W2 = double(net.Layers(13).Weights);
    b2 = double(net.Layers(13).Bias);

    scores = W2 * x + b2(:);

    %% 6. 自行實作 Softmax
    % 先減去最大值，避免 exp 的數值過大。
    scores = scores - max(scores);
    expScores = exp(scores);

    prob = expScores / sum(expScores);
    prob = reshape(prob, 10, 1);
end


function Y = myConv2D(X, layer)
    % 使用圖片小區塊與權重的矩陣乘法，實作卷積層。

    W = double(layer.Weights);
    b = double(layer.Bias);

    stride = double(layer.Stride);
    padding = double(layer.PaddingSize);

    kernelH = size(W, 1);
    kernelW = size(W, 2);
    inputChannels = size(W, 3);
    outputChannels = size(W, 4);

    inputH = size(X, 1);
    inputW = size(X, 2);

    strideH = stride(1);
    strideW = stride(2);

    % PaddingSize 的順序：上、下、左、右
    top = padding(1);
    bottom = padding(2);
    left = padding(3);
    right = padding(4);

    %% 1. 圖片外圍補 0
    paddedX = zeros( ...
        inputH + top + bottom, ...
        inputW + left + right, ...
        inputChannels);

    paddedX(top + (1:inputH), ...
            left + (1:inputW), :) = X;

    %% 2. 計算輸出尺寸
    outputH = floor( ...
        (size(paddedX, 1) - kernelH) / strideH) + 1;

    outputW = floor( ...
        (size(paddedX, 2) - kernelW) / strideW) + 1;

    %% 3. 把每個滑動位置的小區塊攤成一欄
    patchSize = kernelH * kernelW * inputChannels;
    patches = zeros(patchSize, outputH * outputW);

    for channel = 1:inputChannels
        for column = 1:kernelW
            for row = 1:kernelH
                rowIndex = row + ...
                    (column - 1) * kernelH + ...
                    (channel - 1) * kernelH * kernelW;

                rows = row + (0:outputH-1) * strideH;
                columns = column + (0:outputW-1) * strideW;

                values = paddedX(rows, columns, channel);
                patches(rowIndex, :) = values(:).';
            end
        end
    end

    %% 4. 每個濾波器與每個圖片區塊做內積，再加偏置
    weightMatrix = reshape(W, patchSize, outputChannels);

    result = weightMatrix.' * patches + b(:);

    %% 5. 還原成 高度×寬度×通道數
    Y = reshape(result.', outputH, outputW, outputChannels);
end


function Y = myMaxPool(X, layer)
    % 每個池化視窗只保留最大值。

    poolSize = double(layer.PoolSize);
    stride = double(layer.Stride);
    padding = double(layer.PaddingSize);

    inputH = size(X, 1);
    inputW = size(X, 2);
    channels = size(X, 3);

    top = padding(1);
    bottom = padding(2);
    left = padding(3);
    right = padding(4);

    %% 1. 池化補邊使用負無限大，不影響最大值
    paddedX = -inf( ...
        inputH + top + bottom, ...
        inputW + left + right, ...
        channels);

    paddedX(top + (1:inputH), ...
            left + (1:inputW), :) = X;

    %% 2. 計算輸出尺寸
    outputH = floor( ...
        (size(paddedX, 1) - poolSize(1)) / stride(1)) + 1;

    outputW = floor( ...
        (size(paddedX, 2) - poolSize(2)) / stride(2)) + 1;

    Y = -inf(outputH, outputW, channels);

    %% 3. 比較視窗內所有位置，保留最大值
    for row = 1:poolSize(1)
        for column = 1:poolSize(2)
            rows = row + (0:outputH-1) * stride(1);
            columns = column + (0:outputW-1) * stride(2);

            Y = max(Y, paddedX(rows, columns, :));
        end
    end
end