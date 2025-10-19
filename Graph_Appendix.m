%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Figure 9b 复现与敏感性分析图 - 最终版 v2.0
% % (适配于最终的分布概率与标准误数据)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 步骤 0: 环境检查和数据加载
if isempty(ver('curvefit'))
    error('此代码需要 Curve Fitting Toolbox。请检查您的 MATLAB 安装。');
end

% MODIFICATION: 修改为您保存数据的文件名
filename = 'figure_9b_distribution_replication.xlsx'; 
try
    data = readtable(filename);
catch ME
    error('无法读取文件 "%s"。请确保该文件与此脚本在同一目录下。\n错误信息: %s', filename, ME.message);
end

%%%%%%%%%%%%%%%%%%%%%%%%  参数定义区域 %%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODIFICATION: 定义新的Y轴(均值/概率)和误差(标准误)列名
y_cols = {'K11_distribution', 'K9_distribution', 'K13_distribution', 'K27_distribution'};
error_cols = {'K11_prob_sem', 'K9_prob_sem', 'K13_prob_sem', 'K27_prob_sem'}; % 使用标准误作为误差

% MODIFICATION: 更新图例标签和颜色 (4个图)
labels = {'Distribution (K=11)', 'Distribution (K=9)', 'Distribution (K=13)', 'Distribution (K=27)'};
colors = {[238, 199, 159]/255, [116, 182, 159]/255, ...
          [166, 205, 228]/255, [100, 100, 100]/255}; 

% 提取横坐标标签和位置
x_labels = data.RS_bin;
x = (1:height(data))'; % 使用行号作为绘图的x坐标
fontName = 'Arial'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 步骤 1: 创建图形并循环绘图
figure;
set(gcf, 'Position', [100, 100, 1000, 800]);

for i = 1:4
    subplot(2, 2, i);
    
    y = data.(y_cols{i});
    y_error = data.(error_cols{i}); % 获取标准误数据
    
    valid_indices = ~isnan(y) & ~isnan(y_error);
    x_clean = x(valid_indices);
    y_clean = y(valid_indices);
    y_error_clean = y_error(valid_indices);
    
    if length(x_clean) < 3, continue; end
    
    % 1a: 回归分析 (二次多项式拟合)
    [fit_model, gof] = fit(x_clean, y_clean, 'poly2');
    x_fit = linspace(min(x_clean), max(x_clean), 100)';
    % MODIFICATION: 使用标准误来计算预测区间(更接近置信区间)
    % y_pred ± t_value * SEM
    % 为简化，我们仍用predint来示意，但误差棒更关键
    conf_bounds = predint(fit_model, x_fit, 0.95, 'observation', 'off'); 
    R2 = gof.rsquare;
    
    % 计算p值
    n = length(y_clean); k = 2; df_reg = k; df_err = n - k - 1;
    SSR = sum((fit_model(x_clean) - mean(y_clean)).^2);
    MSE = gof.sse / df_err; MSR = SSR / df_reg;
    if MSE > 0, F_statistic = MSR / MSE; p_value = 1 - fcdf(F_statistic, df_reg, df_err); 
    else, p_value = 0; end
    
    % % 1b: 绘图
    % hold on;
    % 
    % % 绘制置信区间阴影
    % fill([x_fit; flipud(x_fit)], [conf_bounds(:,2); flipud(conf_bounds(:,1))], ...
    %     colors{i}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    % 
    % % MODIFICATION: 使用 errorbar 绘制带标准误误差棒的散点
    % errorbar(x_clean, y_clean, y_error_clean, ... % 使用y_error_clean
    %     'o', 'LineStyle', 'none', 'MarkerEdgeColor', 'k', ...
    %     'MarkerFaceColor', colors{i}, 'Color', colors{i}*0.9, ...
    %     'LineWidth', 1, 'CapSize', 5, 'MarkerSize', 7);
    % 1b: 绘图
    hold on;
    
    % 绘制置信区间阴影 (最底层)
    fill([x_fit; flipud(x_fit)], [conf_bounds(:,2); flipud(conf_bounds(:,1))], ...
        colors{i}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        
    % 新增: 使用 errorbar 绘制带标准误误差棒的散点 (中间层)
    errorbar(x_clean, y_clean, y_error_clean, ... % 使用y_error_clean定义误差棒长度
        'o', ...                      % 标记样式
        'LineStyle', 'none', ...      % 不用线连接数据点
        'MarkerEdgeColor', 'k', ...   % 标记点边缘为黑色，更清晰
        'MarkerFaceColor', colors{i}, ... % 标记点填充色
        'Color', colors{i}*0.9, ...   % 误差棒本身的颜色
        'LineWidth', 1, ...           % 误差棒线宽
        'CapSize', 5, ...             % 误差棒顶端“帽子”的大小
        'MarkerSize', 7);             % 标记点大小
        
    % 绘制拟合曲线 (最顶层)
    plot(x_fit, fit_model(x_fit), '-', 'Color', colors{i}*0.7, 'LineWidth', 2.5);
    
    % % 1c: 在图内标记 R² 和 p-值
    % if p_value < 0.001, p_text = 'p < 0.001'; 
    % else, p_text = sprintf('p = %.3f', p_value); end
    % stat_text = {sprintf('R^2 = %.2f', R2), p_text};
    % ax_lims = axis; 
    % text_x = ax_lims(1) + 0.05 * (ax_lims(2) - ax_lims(1));
    % text_y_range = ax_lims(4) - ax_lims(3);
    % text_y = ax_lims(3) + 0.95 * text_y_range; % 定位到左上角
    % text(text_x, text_y, stat_text, 'FontSize', 12, 'FontName', fontName, ...
    %     'VerticalAlignment', 'top'); % 垂直对齐方式改为'top'
    % 1c: 在图内标记 R² 和 p-值 (新版 - 左下角)
    if p_value < 0.001, p_text = 'p < 0.001'; 
    elseif p_value < 0.01, p_text = 'p < 0.01'; 
    elseif p_value < 0.05, p_text = 'p < 0.05'; 
    else, p_text = sprintf('p = %.3f', p_value); end
    stat_text = {sprintf('R^2 = %.2f', R2), p_text};
    
    % 获取坐标轴范围以进行相对定位
    ax_lims = axis; 
    
    % 定义文本在左下角的位置 (x: 距左5%, y: 距下5%)
    text_x = ax_lims(1) + 0.05 * (ax_lims(2) - ax_lims(1));
    text_y = ax_lims(3) + 0.05 * (ax_lims(4) - ax_lims(3)); % 从Y轴底部(ax_lims(3))开始计算
    
    % 放置文本
    text(text_x, text_y, stat_text, ...
        'FontSize', 12, ...            
        'FontName', fontName, ...      
        'VerticalAlignment', 'bottom'); % 对齐方式改为 'bottom'

    



        
    % 1d: 设置坐标轴和标题
    hold off;
    xlabel('Rao-Stirling Index Bin', 'FontSize', 12, 'FontName', fontName);
    ylabel('Disruption (Top 20%)', 'FontSize', 12, 'FontName', fontName);
    title(labels{i}, 'FontSize', 14, 'FontName', fontName);
    set(gca, 'FontSize', 10, 'FontName', fontName);
    xlim([0.5, length(x) + 0.5]);
    xticks(1:2:length(x)); % 每隔一个显示一个刻度
    xticklabels(x_labels(1:2:end));
    xtickangle(45);
    box on;
end