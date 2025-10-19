%% --- 清理工作区和命令窗口 ---
clear;
clc;
close all;

%% --- 参数设置 ---
% (参数设置部分与之前相同)
excelFilePath = 'GraphSAGE_embeddings.xlsx';
journalIdColName = 'JournalID';
clusterColName = 'Cluster';
embeddingStartCol = 3;
numOriginalEmbedDims = 64;
pcaTargetDims = 32;
K = 20;
useMutualKNN = false; % 使用非 Mutual K-NN 以获得更多连接
chunkSize = 1000;
enableSampling = true;
targetNumNodesToPlot = 5000;
minNodeSize = 3;
maxNodeSize = 15;
edgeTransparency = 0.15; % 稍微调高一点透明度
layoutIterations = 750; % 增加迭代次数可能有助于布局

%% --- 1. 加载数据 ---
fprintf('正在从 %s 加载数据...\n', excelFilePath);
% (代码不变)
if ~exist(excelFilePath, 'file'), error('错误: 未找到 Excel 文件: %s', excelFilePath); end
try dataTable = readtable(excelFilePath); fprintf('Excel 文件加载成功。\n');
catch ME, error('加载 Excel 文件时出错: %s', ME.message); end
try
    journalIDs = dataTable.(journalIdColName);
    clusters = dataTable.(clusterColName);
    embeddingEndCol = embeddingStartCol + numOriginalEmbedDims - 1;
    if embeddingEndCol > width(dataTable), error('指定的嵌入维度范围超出表格列数。'); end
    embeddings = dataTable{:, embeddingStartCol:embeddingEndCol};
catch ME, error('从表格提取数据时出错: %s\n请检查列名 (%s, %s) 和范围。', ME.message, journalIdColName, clusterColName); end
if ~isnumeric(clusters) || ~isnumeric(embeddings), error('聚类或嵌入列包含非数值数据。'); end
embeddings = double(embeddings); numNodes = size(embeddings, 1);
fprintf('数据提取完成: %d 个期刊, %d 维原始嵌入。\n', numNodes, size(embeddings, 2));
clear dataTable;

%% --- 2. PCA 降维 ---
fprintf('正在进行 PCA 降维 (%d -> %d 维)...\n', size(embeddings, 2), pcaTargetDims);
% (代码不变)
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
% (代码不变)
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

%% --- 6. 节点抽样 ---
nodes_to_plot_indices = 1:numnodes(G_full); G_plot = G_full; clusters_plot = clusters;
if enableSampling && numnodes(G_full) > targetNumNodesToPlot
    fprintf('启用抽样：目标绘制约 %d 个节点...\n', targetNumNodesToPlot);
    % (抽样代码不变)
    if ~exist('clusters', 'var') || isempty(clusters) || length(clusters) ~= numNodes, error('无法执行抽样，"clusters" 变量无效。'); end
    uniqueClustersPresent = unique(clusters); clusterCounts = accumarray(clusters(:) + 1, 1, [max(clusters)+1, 1]); clusterCounts = clusterCounts(uniqueClustersPresent + 1);
    proportions = clusterCounts / numnodes(G_full); targetSamplesPerCluster = floor(proportions * targetNumNodesToPlot); targetSamplesPerCluster(clusterCounts > 0 & targetSamplesPerCluster == 0) = 1; targetSamplesPerCluster = min(targetSamplesPerCluster, clusterCounts);
    currentTotalSampled = sum(targetSamplesPerCluster); diff = targetNumNodesToPlot - currentTotalSampled;
    if diff ~= 0, fprintf('  调整抽样数量以接近目标值 (差额: %d)...\n', diff); eligible_clusters_indices_in_unique = find(targetSamplesPerCluster > 0);
        if ~isempty(eligible_clusters_indices_in_unique), eligible_cluster_values = uniqueClustersPresent(eligible_clusters_indices_in_unique);
             for k = 1:abs(diff), rand_idx_eligible = randi(length(eligible_cluster_values)); cluster_val_to_adjust = eligible_cluster_values(rand_idx_eligible); idx_in_target = find(uniqueClustersPresent == cluster_val_to_adjust, 1); idx_in_counts = idx_in_target;
                 if diff > 0, if targetSamplesPerCluster(idx_in_target) < clusterCounts(idx_in_counts), targetSamplesPerCluster(idx_in_target) = targetSamplesPerCluster(idx_in_target) + 1; end
                 elseif diff < 0, if targetSamplesPerCluster(idx_in_target) > 1, targetSamplesPerCluster(idx_in_target) = targetSamplesPerCluster(idx_in_target) - 1; end; end
             end; end; end
    finalSampleSizes = targetSamplesPerCluster; sampled_indices_cell = cell(length(uniqueClustersPresent), 1); actualTotalSampled = 0; fprintf('  按簇抽样:\n');
    for i = 1:length(uniqueClustersPresent)
        clusterVal = uniqueClustersPresent(i); numToSample = finalSampleSizes(i);
        if numToSample > 0, nodesInCluster_idx = find(clusters == clusterVal); fprintf('    Cluster %d: 抽样 %d / %d 个节点\n', clusterVal, numToSample, length(nodesInCluster_idx)); numToSample = min(numToSample, length(nodesInCluster_idx));
            if numToSample > 0, perm = randperm(length(nodesInCluster_idx), numToSample); sampled_indices_cell{i} = nodesInCluster_idx(perm); actualTotalSampled = actualTotalSampled + numToSample; end; end; end
    nodes_to_plot_indices = vertcat(sampled_indices_cell{:});
    fprintf('  实际抽样节点总数: %d\n', actualTotalSampled); overallSamplingRatio = actualTotalSampled / numNodes; fprintf('  总体实际抽样比例: %.4f (%.2f%%)\n', overallSamplingRatio, overallSamplingRatio * 100);
    fprintf('  正在创建子图...\n'); G_plot = subgraph(G_full, nodes_to_plot_indices); clusters_plot = clusters(nodes_to_plot_indices);
    fprintf('  子图创建完成: %d 个节点, %d 条边。\n', numnodes(G_plot), numedges(G_plot));
else
    fprintf('未启用抽样或节点数 (%d) 不大于目标数 (%d)，绘制完整图。\n', numnodes(G_full), targetNumNodesToPlot); clusters_plot = clusters;
end
clear G_full sampled_indices_cell finalSampleSizes targetSamplesPerCluster proportions clusterCounts uniqueClustersPresent nodesInCluster_idx perm;

%% --- 6a. 查找并保留最大连通分量 ---
fprintf('正在查找最大连通分量...\n');
G_final = G_plot; % 默认最终图是 G_plot
nodes_in_final_graph_idx = 1:numnodes(G_plot); % 默认所有节点都在最终图中
nodeSizes_final = []; % 初始化
nodeColors_final = []; % 初始化
clusters_final = []; % 初始化

if numnodes(G_plot) > 0 && numedges(G_plot) > 0 % 只在有节点和边时查找
    % 使用 'cell' 输出获取每个分量的节点列表 (索引相对于 G_plot)
    [bin, binsize] = conncomp(G_plot, 'OutputForm', 'cell');

    if ~isempty(binsize)
        % 找到最大分量的索引
        [maxSize, largestCompIdx] = max(cellfun(@numel, bin));
        fprintf('  找到 %d 个连通分量，最大分量包含 %d 个节点。\n', length(bin), maxSize);

        % 获取最大分量中的节点索引 (这些是 G_plot 中的节点索引)
        nodes_in_largest_comp_idx_in_G_plot = bin{largestCompIdx};

        if maxSize < numnodes(G_plot) % 只有当最大分量不是全部节点时才创建子图
             fprintf('  只保留最大连通分量进行绘制...\n');
             % 创建最终的子图
             G_final = subgraph(G_plot, nodes_in_largest_comp_idx_in_G_plot);

             % 记录最终图中的节点索引 (相对于 G_plot)
             nodes_in_final_graph_idx = nodes_in_largest_comp_idx_in_G_plot;
        else
            fprintf('  最大连通分量包含所有节点，无需进一步过滤。\n');
            % G_final 和 nodes_in_final_graph_idx 保持默认值
        end
    else
         fprintf('  未找到连通分量（图可能为空或只有孤立节点），绘制原始 G_plot。\n');
    end
else
     fprintf('  G_plot 没有边或节点，跳过连通分量查找。\n');
end
clear G_plot bin binsize maxSize largestCompIdx nodes_in_largest_comp_idx_in_G_plot; % 清理内存

%% --- 7. 准备最终绘图数据 (颜色, 大小) ---
fprintf('准备最终绘图节点属性...\n');
% --- 过滤节点属性以匹配 G_final ---
% clusters_plot 包含 G_plot 中所有节点（抽样后）的聚类信息
% nodeColors_plot 和 nodeSizes_plot 是根据 G_plot 计算的

% 从 G_plot 的属性中，只选择属于 G_final 的节点 (由 nodes_in_final_graph_idx 指定)
if ~isempty(nodes_in_final_graph_idx)
    % clusters_final: 获取最终图中节点的聚类标签
    clusters_final = clusters_plot(nodes_in_final_graph_idx);

    % nodeColors_final: 计算最终图节点的颜色
    if ~isempty(clusters_final)
        uniqueClusters_final = unique(clusters_final); minCluster_final = min(clusters_final); maxCluster_final = max(clusters_final);
        if minCluster_final < 0 || maxCluster_final > 10, warning('最终图节点的聚类标签范围异常。'); end
        if isempty(clusters_final), numColorsNeeded_final = 1; else numColorsNeeded_final = max(clusters_final) + 1; end
        cmap_final = turbo(numColorsNeeded_final);
        colorIndices_final = max(1, clusters_final + 1); colorIndices_final(colorIndices_final > numColorsNeeded_final) = numColorsNeeded_final;
        nodeColors_final = cmap_final(colorIndices_final, :);
    else
        nodeColors_final = []; % 如果没有节点，颜色为空
    end

    % nodeSizes_final: 计算最终图节点的度中心性大小
    fprintf('计算最终图节点的度中心性...\n');
    if numedges(G_final) > 0
        nodeDegrees_final = degree(G_final);
    else
        nodeDegrees_final = zeros(numnodes(G_final), 1);
    end
    minDeg_final = min(nodeDegrees_final); maxDeg_final = max(nodeDegrees_final);
    nodeSizes_final = zeros(numnodes(G_final), 1);
    if maxDeg_final == minDeg_final
        nodeSizes_final(:) = mean([minNodeSize, maxNodeSize]);
    else
        nodeSizes_final = minNodeSize + (nodeDegrees_final - minDeg_final) * (maxNodeSize - minNodeSize) / (maxDeg_final - minDeg_final);
    end
    nodeSizes_final(isnan(nodeSizes_final)) = minNodeSize;
    nodeSizes_final(nodeDegrees_final == 0) = minNodeSize; % 理论上这里度数都 > 0

else
    fprintf('警告: 最终图中没有节点。\n');
    nodeSizes_final = [];
    nodeColors_final = [];
    clusters_final = [];
end

% 清理 G_plot 相关的变量
clear clusters_plot nodeDegrees_final minDeg_final maxDeg_final uniqueClusters_final minCluster_final maxCluster_final numColorsNeeded_final cmap_final colorIndices_final;

%% --- 8. 绘制网络图 (绘制 G_final) ---
fprintf('正在绘制网络图 (最大连通分量)...\n');
figure('Name', '期刊相似性网络图 (K-NN, 最大连通分量)', 'NumberTitle', 'off', 'WindowState', 'maximized');

if isempty(G_final) || numnodes(G_final) == 0
     fprintf('最终图为空，无法绘制。\n');
else
    plotOptions = {'Layout', 'force', 'Iterations', layoutIterations};
    if numedges(G_final) > 0
        plotOptions = [plotOptions, {'WeightEffect', 'inverse'}];
    else
        plotOptions = [plotOptions, {'WeightEffect', 'none'}];
    end

    h = plot(G_final, plotOptions{:});

    % --- 自定义节点外观 (使用 _final 变量) ---
    if ~isempty(nodeSizes_final)
        h.MarkerSize = nodeSizes_final;
    end
    if ~isempty(nodeColors_final)
        h.NodeColor = nodeColors_final; % 直接赋 RGB
    end

    % --- 自定义边外观 ---
    if numedges(G_final) > 0
        h.EdgeAlpha = edgeTransparency;
        h.EdgeColor = [0.3 0.3 0.3];
    end

    % --- 节点标签 (默认不显示) ---
    h.NodeLabel = {};

    % --- 添加标题和坐标轴设置 ---
    % titleStr = sprintf('期刊相似性网络图 (K-NN, K=%d, 最大连通分量)', K);
    % originalSampledCount = length(nodes_to_plot_indices); % 获取抽样后的节点数
    % if enableSampling && originalSampledCount < numNodes
    %     titleStr = [titleStr, sprintf('\n(基于 %d / %d 抽样节点)', numnodes(G_final), originalSampledCount)];
    % end
    % title({titleStr, sprintf('(PCA %dD, Size: Degree, Color: Cluster)', pcaTargetDims)});
    % axis off; set(gca, 'LooseInset', get(gca, 'TightInset'));

    %% --- 9. 添加图例 (基于 G_final 中的聚类) ---
    fprintf('正在添加图例...\n');
    if exist('clusters_final', 'var') && ~isempty(clusters_final)
        presentClusters_final = unique(clusters_final);
        presentClusters_final = sort(presentClusters_final(presentClusters_final>=0));
        if ~isempty(presentClusters_final)
            legendLabels = cell(length(presentClusters_final), 1); legendColors = zeros(length(presentClusters_final), 3);
            if isempty(clusters_final), cmapMax = 1; else cmapMax = max(clusters_final) + 1; end
            cmap_for_legend = turbo(cmapMax); validLegendEntries = 0;
            for i = 1:length(presentClusters_final)
                clusterVal = presentClusters_final(i); validLegendEntries = validLegendEntries + 1;
                legendLabels{validLegendEntries} = sprintf('Cluster %d', clusterVal);
                colorIdx = max(1, clusterVal + 1); colorIdx = min(colorIdx, size(cmap_for_legend, 1));
                legendColors(validLegendEntries, :) = cmap_for_legend(colorIdx, :);
            end
            legendLabels = legendLabels(1:validLegendEntries); legendColors = legendColors(1:validLegendEntries, :);
            if ~isempty(legendLabels) && validLegendEntries <= 15
                hold on; legendHandles = gobjects(validLegendEntries, 1);
                for i = 1:validLegendEntries, legendHandles(i) = scatter(NaN, NaN, 50, legendColors(i,:), 'filled', 'DisplayName', legendLabels{i}, 'MarkerEdgeColor', 'k'); end
                hold off; lgd = legend(legendHandles, 'Location', 'eastoutside', 'FontSize', 12);
                if ~isempty(lgd) && isvalid(lgd), lgd.Title.String = 'Cluster'; end % 正确设置标题
            else fprintf('最终图中聚类数量 (%d) 为 0 或过多 (>15)，未自动添加图例。\n', validLegendEntries); end
        else
            fprintf('最终图中未找到有效的聚类标签，无法添加图例。\n');
        end
    else
         fprintf('警告: 无法获取最终图节点的聚类信息 (clusters_final)，无法添加图例。\n');
    end
    clear clusters_final presentClusters_final legendLabels legendColors cmap_for_legend validLegendEntries legendHandles lgd;
end % 结束 if isempty(G_final)

fprintf('--- 处理完成 ---.\n');