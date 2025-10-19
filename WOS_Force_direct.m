%% --- 清理工作区和命令窗口 ---
clear;
clc;
close all;

%% --- 参数设置 ---
excelFilePath = 'GraphSAGE_embeddings_WOS.xlsx'; % <<--- 修改: 文件路径
% journalIdColName = 'JournalID'; % 不再需要 JournalID 列名
categoryColName = 'WOS_Category';     % <<--- 修改: WoS 类别列名
embeddingStartCol = 3;                 % <<--- 确认: 嵌入向量从第 3 列开始
numOriginalEmbedDims = 64;             % 原始嵌入维度
pcaTargetDims = 32;                    % PCA 降维后的目标维度

% --- K-NN 图参数 ---
K = 20;
useMutualKNN = false; % 使用非 Mutual 以获得更多连接

% --- 性能与内存相关参数 ---
chunkSize = 1000;

% --- 抽样参数 ---
enableSampling = true;
targetNumNodesToPlot = 5000;

% --- 外观参数 ---
minNodeSize = 3;
maxNodeSize = 15;
edgeTransparency = 0.15; % 可以根据边的密度调整
layoutIterations = 750;

%% --- 1. 加载数据 ---
fprintf('正在从 %s 加载数据...\n', excelFilePath);
if ~exist(excelFilePath, 'file'), error('错误: 未找到 Excel 文件: %s', excelFilePath); end
try dataTable = readtable(excelFilePath); fprintf('Excel 文件加载成功。\n');
catch ME, error('加载 Excel 文件时出错: %s', ME.message); end

try
    % 读取类别名称 (可能是 cellstr 或 categorical)
    categories = dataTable.(categoryColName);
    % 确保是 cell array of strings 或 categorical
    if iscell(categories) && ~iscellstr(categories) % R2016b 之前没有 iscellstr
        % 尝试转换非字符串 cell 内容为字符串
        categories = cellfun(@string, categories, 'UniformOutput', false);
    elseif ~iscell(categories) && ~iscategorical(categories)
        warning('类别列 "%s" 不是 cell 或 categorical 类型，尝试转换为字符串。', categoryColName);
        categories = string(categories); % 转换为 string array
    end
    % 如果是 categorical，后面处理时可以直接使用
    if iscategorical(categories)
         fprintf('类别列为 categorical 类型。\n');
    elseif iscell(categories)
         fprintf('类别列为 cell 类型。\n');
    else % string array (R2016b+)
         fprintf('类别列为 string 类型。\n');
    end


    % 读取嵌入向量
    embeddingEndCol = embeddingStartCol + numOriginalEmbedDims - 1;
    if embeddingEndCol > width(dataTable), error('指定的嵌入维度范围超出表格列数。'); end
    embeddings = dataTable{:, embeddingStartCol:embeddingEndCol};
catch ME
     error('从表格提取数据时出错: %s\n请检查列名 (%s) 和嵌入列范围是否正确。', ME.message, categoryColName);
end
if ~isnumeric(embeddings), error('嵌入列包含非数值数据。'); end
embeddings = double(embeddings);
numNodes = size(embeddings, 1);
fprintf('数据提取完成: %d 个记录, %d 维原始嵌入。\n', numNodes, size(embeddings, 2));
clear dataTable;

%% --- 2. PCA 降维 ---
fprintf('正在进行 PCA 降维 (%d -> %d 维)...\n', size(embeddings, 2), pcaTargetDims);
% (PCA 代码不变)
if size(embeddings, 2) < pcaTargetDims, reducedEmbeddings = embeddings;
else
    try [~, score, ~, ~, explained] = pca(embeddings, 'NumComponents', pcaTargetDims); reducedEmbeddings = score;
        fprintf('PCA 完成。保留的前 %d 个主成分解释了 %.2f%% 的方差。\n', pcaTargetDims, sum(explained(1:pcaTargetDims)));
    catch ME, error('PCA 计算失败: %s', ME.message); end
end
fprintf('降维后维度: %d x %d\n', size(reducedEmbeddings, 1), size(reducedEmbeddings, 2));
clear embeddings score explained;

%% --- 3. 优化：分块计算相似性并确定 K-NN ---
fprintf('开始分块计算相似性并确定 Top-%d 邻居 (Chunk Size: %d)...\n', K, chunkSize);
% (K-NN 计算代码不变)
fprintf('  正在归一化嵌入向量...\n'); normEmbeddings = reducedEmbeddings ./ max(1e-9, vecnorm(reducedEmbeddings, 2, 2)); clear reducedEmbeddings; fprintf('  归一化完成。\n');
topKData = cell(numNodes, 1); neighborCount = zeros(numNodes, 1); minSimilarityInK = -inf(numNodes, 1); numChunks = ceil(numNodes / chunkSize); fprintf('  总共 %d 个块需要处理。\n', numChunks); totalComparisons = numChunks * (numChunks + 1) / 2; comparisonsDone = 0; tic; fprintf('  处理块对 (寻找潜在 Top-K):\n');
for i = 1:numChunks
    startIndex_i = (i - 1) * chunkSize + 1; endIndex_i = min(i * chunkSize, numNodes); chunk_i_indices = startIndex_i:endIndex_i; if ~isscalar(startIndex_i) || ~isreal(startIndex_i) || ~isscalar(endIndex_i) || ~isreal(endIndex_i), error('索引错误 i=%d',i); end; chunk_i = normEmbeddings(chunk_i_indices, :);
    for j = i:numChunks
        startIndex_j = (j - 1) * chunkSize + 1; endIndex_j = min(j * chunkSize, numNodes); chunk_j_indices = startIndex_j:endIndex_j; if ~isscalar(startIndex_j) || ~isreal(startIndex_j) || ~isscalar(endIndex_j) || ~isreal(endIndex_j), error('索引错误 j=%d',j); end; chunk_j = normEmbeddings(chunk_j_indices, :); chunkSim = chunk_i * chunk_j';
        numRows = size(chunkSim, 1); numCols = size(chunkSim, 2);
        for row = 1:numRows
            global_i = chunk_i_indices(row); startCol = 1; if i == j, startCol = row + 1; end
            for col = startCol:numCols
                global_j = chunk_j_indices(col); similarity = chunkSim(row, col);
                if neighborCount(global_i) < K, topKData{global_i}(end+1, :) = [similarity, global_j]; neighborCount(global_i)=neighborCount(global_i)+1; if neighborCount(global_i) == K, minSimilarityInK(global_i) = min(topKData{global_i}(:, 1)); end
                elseif similarity > minSimilarityInK(global_i), [~, minIdx] = min(topKData{global_i}(:, 1)); topKData{global_i}(minIdx, :) = [similarity, global_j]; minSimilarityInK(global_i) = min(topKData{global_i}(:, 1)); end
                if neighborCount(global_j) < K, topKData{global_j}(end+1, :) = [similarity, global_i]; neighborCount(global_j)=neighborCount(global_j)+1; if neighborCount(global_j) == K, minSimilarityInK(global_j) = min(topKData{global_j}(:, 1)); end
                elseif similarity > minSimilarityInK(global_j), [~, minIdx] = min(topKData{global_j}(:, 1)); topKData{global_j}(minIdx, :) = [similarity, global_i]; minSimilarityInK(global_j) = min(topKData{global_j}(:, 1)); end
            end; end; clear chunkSim chunk_j; comparisonsDone = comparisonsDone + 1;
        if mod(comparisonsDone, max(1, floor(totalComparisons / 20))) == 0 || comparisonsDone == totalComparisons, elapsedTime = toc; percentDone = comparisonsDone / totalComparisons * 100; fprintf('  进度: %.1f%% (%d / %d 块对), 已耗时 %.1f 秒\n', percentDone, comparisonsDone, totalComparisons, elapsedTime); end
    end; clear chunk_i;
end; clear normEmbeddings minSimilarityInK neighborCount; fprintf('Top-K 邻居候选确定完成。\n');

%% --- 4. 构建最终边列表 ---
fprintf('正在构建最终边列表 (K=%d, Mutual=%d)...\n', K, useMutualKNN);
% (代码不变)
initialCapacity = round(numNodes * K); if ~useMutualKNN, initialCapacity = initialCapacity * 2; end; edgeSource = zeros(initialCapacity, 1); edgeTarget = zeros(initialCapacity, 1); edgeSim = zeros(initialCapacity, 1); edgeCount = 0;
for i = 1:numNodes
    if isempty(topKData{i}), continue; end; neighbors_i = topKData{i}(:, 2); similarities_i = topKData{i}(:, 1);
    for idx = 1:length(neighbors_i)
        j = neighbors_i(idx); sim_ij = similarities_i(idx); addEdge = false;
        if useMutualKNN, if i < j, if ~isempty(topKData{j}) && ismember(i, topKData{j}(:, 2)), addEdge = true; end; end
        else addEdge = true; end
        if addEdge, edgeCount = edgeCount + 1;
            if edgeCount > length(edgeSource), newCapacity = round(length(edgeSource) * 1.5); edgeSource = [edgeSource; zeros(newCapacity - length(edgeSource), 1)]; edgeTarget = [edgeTarget; zeros(newCapacity - length(edgeTarget), 1)]; edgeSim = [edgeSim; zeros(newCapacity - length(edgeSim), 1)]; initialCapacity = length(edgeSource); fprintf('  扩容边列表至 %d...\n', initialCapacity); end
            edgeSource(edgeCount) = i; edgeTarget(edgeCount) = j; edgeSim(edgeCount) = sim_ij; end
    end; end
edgeSource = edgeSource(1:edgeCount); edgeTarget = edgeTarget(1:edgeCount); edgeSim = edgeSim(1:edgeCount);
if ~useMutualKNN, fprintf('  合并非 Mutual K-NN 的对称边...\n'); edge_list_undir = [min(edgeSource, edgeTarget), max(edgeSource, edgeTarget)]; [~, unique_idx] = unique(edge_list_undir, 'rows', 'first'); edgeSource = edgeSource(unique_idx); edgeTarget = edgeTarget(unique_idx); edgeSim = edgeSim(unique_idx); fprintf('  合并后剩余 %d 条唯一无向边。\n', length(edgeSource)); end
edge_list = [edgeSource, edgeTarget]; edge_weights = 1.0 - edgeSim; edge_weights(edge_weights < 1e-6) = 1e-6;
fprintf('最终构建了 %d 条边。\n', size(edge_list, 1));
clear topKData edgeSource edgeTarget edgeSim edge_list_undir unique_idx;

%% --- 5. 创建图对象 ---
fprintf('正在创建图对象...\n');
% (代码不变)
if ~isempty(edge_list), G_full = graph(edge_list(:,1), edge_list(:,2), edge_weights, numNodes);
else G_full = graph([], [], [], numNodes); end
fprintf('完整图对象创建完成: %d 个节点, %d 条边。\n', numnodes(G_full), numedges(G_full));
clear edge_list edge_weights;

%% --- 6. 节点抽样 (基于 Category Name) ---
nodes_to_plot_indices = 1:numnodes(G_full); G_plot = G_full; categories_plot = categories; % 默认情况
if enableSampling && numnodes(G_full) > targetNumNodesToPlot
    fprintf('启用抽样：目标绘制约 %d 个节点...\n', targetNumNodesToPlot);
    if ~exist('categories', 'var') || isempty(categories) || length(categories) ~= numNodes
        error('无法执行抽样，因为 "categories" 变量无效或长度不匹配。');
    end

    % --- 修改：基于类别名称进行统计和抽样 ---
    [uniqueCategories, ~, group_idx] = unique(categories); % 获取唯一类别和每个节点的组索引
    numUniqueCategories = length(uniqueCategories);

    % 计算每个类别的节点数
    categoryCounts = accumarray(group_idx, 1, [numUniqueCategories, 1]);

    proportions = categoryCounts / numnodes(G_full);
    targetSamplesPerCategory = floor(proportions * targetNumNodesToPlot);
    targetSamplesPerCategory(categoryCounts > 0 & targetSamplesPerCategory == 0) = 1; % 至少抽 1 个
    targetSamplesPerCategory = min(targetSamplesPerCategory, categoryCounts); % 不超过实际数量

    currentTotalSampled = sum(targetSamplesPerCategory);
    diff = targetNumNodesToPlot - currentTotalSampled;
    if diff ~= 0 % 调整逻辑
        fprintf('  调整抽样数量以接近目标值 (差额: %d)...\n', diff);
        eligible_categories_indices = find(targetSamplesPerCategory > 0); % 在 uniqueCategories 中的索引
        if ~isempty(eligible_categories_indices)
             for k = 1:abs(diff)
                 rand_idx_eligible = randi(length(eligible_categories_indices));
                 category_index_to_adjust = eligible_categories_indices(rand_idx_eligible); % 在 uniqueCategories/targetSamples/counts 中的索引

                 if diff > 0
                     if targetSamplesPerCategory(category_index_to_adjust) < categoryCounts(category_index_to_adjust)
                         targetSamplesPerCategory(category_index_to_adjust) = targetSamplesPerCategory(category_index_to_adjust) + 1;
                     end
                 elseif diff < 0
                     if targetSamplesPerCategory(category_index_to_adjust) > 1
                         targetSamplesPerCategory(category_index_to_adjust) = targetSamplesPerCategory(category_index_to_adjust) - 1;
                     end
                 end
             end
        end
    end

    finalSampleSizes = targetSamplesPerCategory;
    sampled_indices_cell = cell(numUniqueCategories, 1);
    actualTotalSampled = 0;
    fprintf('  按类别抽样:\n');
    for i = 1:numUniqueCategories
        categoryName = uniqueCategories(i);
        numToSample = finalSampleSizes(i);
        if numToSample > 0
            % 找到属于这个类别的所有节点的原始索引
            if iscategorical(categories)
                nodesInCat_idx = find(categories == categoryName);
            else % cellstr or string array
                 nodesInCat_idx = find(strcmp(categories, categoryName)); % 使用 strcmp for cell/string
            end

            % 显示时可能需要处理特殊字符
            if iscell(categoryName), categoryNameDisp = categoryName{1}; else categoryNameDisp = string(categoryName); end
            fprintf('    Category "%s": 抽样 %d / %d 个节点\n', categoryNameDisp, numToSample, length(nodesInCat_idx));

            numToSample = min(numToSample, length(nodesInCat_idx));
            if numToSample > 0
                perm = randperm(length(nodesInCat_idx), numToSample);
                sampled_indices_cell{i} = nodesInCat_idx(perm);
                actualTotalSampled = actualTotalSampled + numToSample;
            end
        end
    end
    nodes_to_plot_indices = vertcat(sampled_indices_cell{:});
    fprintf('  实际抽样节点总数: %d\n', actualTotalSampled);
    overallSamplingRatio = actualTotalSampled / numNodes; fprintf('  总体实际抽样比例: %.4f (%.2f%%)\n', overallSamplingRatio, overallSamplingRatio * 100);
    fprintf('  正在创建子图...\n');
    G_plot = subgraph(G_full, nodes_to_plot_indices);
    categories_plot = categories(nodes_to_plot_indices); % 获取子图节点的类别信息
    fprintf('  子图创建完成: %d 个节点, %d 条边。\n', numnodes(G_plot), numedges(G_plot));
else
    fprintf('未启用抽样或节点数 (%d) 不大于目标数 (%d)，绘制完整图。\n', numnodes(G_full), targetNumNodesToPlot);
    categories_plot = categories; % 未抽样时，绘图类别即为原始类别
end
clear G_full sampled_indices_cell finalSampleSizes targetSamplesPerCategory proportions categoryCounts uniqueCategories group_idx nodesInCat_idx perm;

%% --- 6a. 查找并保留最大连通分量 ---
fprintf('正在查找最大连通分量...\n');
% (代码不变)
G_final = G_plot; nodes_in_final_graph_idx = 1:numnodes(G_plot); nodeSizes_final = []; nodeColors_final = []; categories_final = []; % 初始化
if numnodes(G_plot) > 0 && numedges(G_plot) > 0
    [bin, binsize] = conncomp(G_plot, 'OutputForm', 'cell');
    if ~isempty(binsize)
        [maxSize, largestCompIdx] = max(cellfun(@numel, bin)); fprintf('  找到 %d 个连通分量，最大分量包含 %d 个节点。\n', length(bin), maxSize);
        nodes_in_largest_comp_idx_in_G_plot = bin{largestCompIdx};
        if maxSize < numnodes(G_plot), fprintf('  只保留最大连通分量进行绘制...\n'); G_final = subgraph(G_plot, nodes_in_largest_comp_idx_in_G_plot); nodes_in_final_graph_idx = nodes_in_largest_comp_idx_in_G_plot;
        else fprintf('  最大连通分量包含所有节点，无需进一步过滤。\n'); end
    else fprintf('  未找到连通分量，绘制原始 G_plot。\n'); end
else fprintf('  G_plot 没有边或节点，跳过连通分量查找。\n'); end
clear G_plot bin binsize maxSize largestCompIdx nodes_in_largest_comp_idx_in_G_plot;

%% --- 7. 准备最终绘图数据 (颜色, 大小) ---
fprintf('准备最终绘图节点属性...\n');
if ~isempty(nodes_in_final_graph_idx)
    % --- 修改：基于 Category Name 分配颜色 ---
    categories_final = categories_plot(nodes_in_final_graph_idx); % 获取最终图节点的类别
    if ~isempty(categories_final)
        [uniqueCategories_final, ~, category_idx_final] = unique(categories_final); % 获取唯一类别和索引
        numUniqueCategories_final = length(uniqueCategories_final);
        cmap_final = turbo(numUniqueCategories_final); % 为实际存在的类别创建颜色图
        nodeColors_final = cmap_final(category_idx_final, :); % 使用索引分配颜色
    else
        nodeColors_final = [];
    end

    % --- 节点大小计算 (不变) ---
    fprintf('计算最终图节点的度中心性...\n');
    if numedges(G_final) > 0, nodeDegrees_final = degree(G_final); else nodeDegrees_final = zeros(numnodes(G_final), 1); end
    minDeg_final = min(nodeDegrees_final); maxDeg_final = max(nodeDegrees_final); nodeSizes_final = zeros(numnodes(G_final), 1);
    if maxDeg_final == minDeg_final, nodeSizes_final(:) = mean([minNodeSize, maxNodeSize]);
    else nodeSizes_final = minNodeSize + (nodeDegrees_final - minDeg_final) * (maxNodeSize - minNodeSize) / (maxDeg_final - minDeg_final); end
    nodeSizes_final(isnan(nodeSizes_final)) = minNodeSize; nodeSizes_final(nodeDegrees_final == 0) = minNodeSize;
else
    fprintf('警告: 最终图中没有节点。\n'); nodeSizes_final = []; nodeColors_final = []; categories_final = []; end
clear categories_plot nodeDegrees_final minDeg_final maxDeg_final category_idx_final cmap_final; % 清理 G_plot 相关的

%% --- 8. 绘制网络图 (绘制 G_final) ---
fprintf('正在绘制网络图 (最大连通分量)...\n');
figure('Name', 'WoS Category Similarity Network (K-NN, Largest Component)', 'NumberTitle', 'off', 'WindowState', 'maximized'); % <<--- 修改: 窗口标题

if isempty(G_final) || numnodes(G_final) == 0, fprintf('最终图为空，无法绘制。\n');
else
    plotOptions = {'Layout', 'force', 'Iterations', layoutIterations};
    if numedges(G_final) > 0, plotOptions = [plotOptions, {'WeightEffect', 'inverse'}]; else plotOptions = [plotOptions, {'WeightEffect', 'none'}]; end
    h = plot(G_final, plotOptions{:});
    if ~isempty(nodeSizes_final), h.MarkerSize = nodeSizes_final; end
    if ~isempty(nodeColors_final), h.NodeColor = nodeColors_final; end % 直接赋 RGB
    if numedges(G_final) > 0, h.EdgeAlpha = edgeTransparency; h.EdgeColor = [0.3 0.3 0.3]; end
    h.NodeLabel = {};
    titleStr = sprintf('WoS Category Similarity Network (K-NN, K=%d, Largest Component)', K); % <<--- 修改: 绘图标题
    originalSampledCount = length(nodes_to_plot_indices);
    if enableSampling && originalSampledCount < numNodes, titleStr = [titleStr, sprintf('\n(Based on %d / %d Sampled Nodes)', numnodes(G_final), originalSampledCount)]; end
    title({titleStr, sprintf('(PCA %dD, Size: Degree, Color: Category)', pcaTargetDims)}); % <<--- 修改: 标题细节
    axis off; set(gca, 'LooseInset', get(gca, 'TightInset'));

    %% --- 9. 添加图例 (基于 Category Name) ---
    fprintf('正在添加图例...\n');
    lgd = []; legendHandles = []; % 初始化
    if exist('categories_final', 'var') && ~isempty(categories_final) % 检查最终类别的变量是否存在
        [presentCategories_final, ~, category_idx_final] = unique(categories_final); % 再次获取唯一类别和索引
        presentCategories_final = sort(presentCategories_final); % 对类别名称排序

        if ~isempty(presentCategories_final)
            numUniqueFinal = length(presentCategories_final);
            legendLabels = cell(numUniqueFinal, 1);
            legendColors = zeros(numUniqueFinal, 3);
            cmap_for_legend = turbo(numUniqueFinal); % 为图例创建颜色图

            validLegendEntries = 0;
            for i = 1:numUniqueFinal
                 categoryName = presentCategories_final(i);
                 validLegendEntries = validLegendEntries + 1;
                 % 处理显示名称 (如果需要)
                 if iscell(categoryName), categoryNameDisp = categoryName{1}; else categoryNameDisp = string(categoryName); end
                 legendLabels{validLegendEntries} = categoryNameDisp;
                 % 找到这个类别在 unique 列表中的索引 i，直接用 i 从 cmap 取色
                 legendColors(validLegendEntries, :) = cmap_for_legend(i, :);
            end
            legendLabels = legendLabels(1:validLegendEntries);
            legendColors = legendColors(1:validLegendEntries, :);

            if ~isempty(legendLabels) && validLegendEntries > 0 && validLegendEntries <= 25 % <<--- 增加图例显示数量限制
                hold on; legendHandles = gobjects(validLegendEntries, 1);
                for i = 1:validLegendEntries
                    legendHandles(i) = scatter(NaN, NaN, 50, legendColors(i,:), 'filled', 'DisplayName', legendLabels{i}, 'MarkerEdgeColor', 'k');
                end
                hold off;
                try
                    lgd = legend(legendHandles, 'Location', 'eastoutside', 'FontSize', 12, 'Interpreter', 'none'); % 关闭解释器防止特殊字符问题
                    if ~isempty(lgd) && isvalid(lgd), lgd.Title.String = 'WoS Category'; end % <<--- 修改: 图例标题
                catch ME_legend
                    fprintf('创建或设置图例时出错: %s\n', ME_legend.message);
                end
            else
                fprintf('最终图中类别数量 (%d) 为 0 或过多 (>25)，未自动添加图例。\n', validLegendEntries);
            end
        else
             fprintf('最终图中未找到有效的类别标签，无法添加图例。\n');
        end
    else
         fprintf('警告: 无法获取最终图节点的类别信息 (categories_final)，无法添加图例。\n');
    end
    clear categories_final presentCategories_final legendLabels legendColors cmap_for_legend validLegendEntries legendHandles lgd category_idx_final;
end % 结束 if isempty(G_final)

fprintf('--- 处理完成 ---.\n');