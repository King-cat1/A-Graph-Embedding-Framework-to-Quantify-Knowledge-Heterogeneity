% % 第一张组合图
% % 加载实际数据
% filename = 'RS_analysis.xlsx';
% data = readtable(filename, 'Sheet', '');
% 
% % 字段信息
% fields = {'Citation_Count', 'Disruption', 'Atyp_Median_Z', 'SB_B', ...
%           'SB_T', 'WSB_mu', 'WSB_sigma', 'WSB_Cinf'};
% labels = {'Citation', 'Disruption', 'Z-Median', 'Sleeping Beauty Coefficient', ...
%           'Awakening Time', 'Immediacy', 'Longevity', 'Ultimate Impact'};
% colors = {[238, 199, 159]/255, [238, 199, 159]/255, ...
%           [241, 223, 164]/255, [241, 223, 164]/255, ...
%           [116, 182, 159]/255, [116, 182, 159]/255, ...
%           [166, 205, 228]/255, [166, 205, 228]/255};
% 
% % 检查并将 Percentile 转换为数值类型
% x = data.Percentile; % 横坐标
% 
% % 创建子图
% figure;
% for i = 1:length(fields)
%     subplot(2, 4, i); % 创建2行4列子图
%     y = data.(fields{i}); % 纵坐标
% 
%     % 去除缺失值
%     valid_idx = ~isnan(y);
%     x_valid = x(valid_idx);
%     y_valid = y(valid_idx);
% 
%     % 绘制散点图（空心）
%     scatter(x_valid, y_valid, 80, 'o', 'MarkerEdgeColor', colors{i}, ...
%         'MarkerFaceColor', 'none', 'LineWidth', 1.5);
%     hold on;
% 
%     % 检查数据点是否足够进行拟合
%     if length(x_valid) > 2
%         % 拟合二次多项式曲线
%         p = polyfit(x_valid, y_valid, 2);
%         x_fit = linspace(min(x_valid), max(x_valid), 100);
%         y_fit = polyval(p, x_fit);
% 
%         % 绘制拟合曲线
%         plot(x_fit, y_fit, '-', 'Color', colors{i}, 'LineWidth', 2);
%     end
% 
%     % 设置坐标轴和标题
%     xlabel('Percentile (%)', 'FontSize', 12);
%     ylabel(labels{i}, 'FontSize', 12);
%     set(gca, 'FontSize', 10);
%     xlim([0, 100]); % 限制横轴范围
% end
% 
% % 保存图像为高分辨率 (DPI=600)
% output_file = 'Field_wise_Analysis.png'; % 输出文件名
% print(gcf, output_file, '-dpng', '-r600'); % 保存为 PNG 格式，分辨率为 600 DPI
% 
% % 调整布局
% % sgtitle('Field-wise Analysis', 'FontSize', 14); % 设置总标题
% set(gcf, 'Position', [100, 100, 1400, 700]); % 调整图形大小


%第二张组合图

% 加载数据
filename = 'paper_top_bottom_analysis_results_20250206_153457.xlsx';
data = readtable(filename, 'Sheet', 2);

% 字段信息
fields = {'Citation_Count', 'Disruption', 'Atyp_Median_Z', 'SB_B', ...
          'SB_T', 'WSB_mu', 'WSB_sigma', 'WSB_Cinf'};
labels = {'Citation', 'Disruption', 'Z-Median', 'Sleeping Beauty Coefficient', ...
          'Awakening Time', 'Immediacy', 'Longevity', 'Ultimate Impact'};
colors = {[238, 199, 159]/255, [238, 199, 159]/255, ...
          [241, 223, 164]/255, [241, 223, 164]/255, ...
          [116, 182, 159]/255, [116, 182, 159]/255, ...
          [166, 205, 228]/255, [166, 205, 228]/255};

% 提取横坐标 (RS_bin 转换为数值)
x_labels = data.RS_bin; % 原始分bin标签
x = 1:height(data); % 将 bin 转换为序数索引

% 创建组合图
figure;
for i = 1:length(fields)
    subplot(2, 4, i); % 创建2行4列子图
    y = data.(fields{i}); % 纵坐标

    % 绘制散点图（空心）
    scatter(x, y, 80, 'o', 'MarkerEdgeColor', colors{i}, ...
        'MarkerFaceColor', 'none', 'LineWidth', 1.5);
    hold on;

    % 拟合曲线（这里使用线性拟合作为示例，可根据实际需求调整）
    p = polyfit(x, y, 2); % 二次多项式拟合
    x_fit = linspace(min(x), max(x), 100);
    y_fit = polyval(p, x_fit);

    % 绘制拟合曲线
    plot(x_fit, y_fit, '-', 'Color', colors{i}, 'LineWidth', 2);

    % 设置坐标轴和标题
    xlabel('Rao-Stirling Index Bin', 'FontSize', 12);
    ylabel([labels{i}, ' (Bottom20%)'], 'FontSize', 12); % 在Y轴标题后添加 "(Top20%)"
    set(gca, 'FontSize', 10);
    xlim([0.5, length(x) + 0.5]); % 限制横轴范围

    % 调整 X 轴刻度标签显示
    xticks(1:3:length(x)); % 每隔 3 个显示一个标签
    xticklabels(x_labels(1:3:end)); % 对应的 bin 标签
    xtickangle(45); % 标签旋转角度为 30 度
end

% 调整布局
% sgtitle('RS Bin Analysis Across Metrics', 'FontSize', 14); % 设置总标题
set(gcf, 'Position', [100, 100, 1400, 700]); % 调整图形大小


% % 加载数据
% filename = 'paper_top_bottom_analysis_results_20250206_153457.xlsx';
% 
% % 读取 Top 20% 和 Bottom 20% 的数据
% data_top = readtable(filename, 'Sheet', 'Top Papers (Across Bins)');
% data_bottom = readtable(filename, 'Sheet', 'Bottom Papers (Across Bins)');
% 
% % 提取横坐标 (RS_bin 作为分类变量)
% x_labels = data_top.RS_bin; % 原始 bin 标签
% x = 1:height(data_top); % 用 1, 2, 3, ... 代替 bin 编号
% 
% % 定义要分析的字段
% fields = {'Citation_Count', 'Disruption', 'Atyp_Median_Z', 'SB_B', ...
%           'SB_T', 'WSB_mu', 'WSB_sigma', 'WSB_Cinf'};
% labels = {'Citation', 'Disruption', 'Z-Median', 'Sleeping Beauty Coefficient', ...
%           'Awakening Time', 'Immediacy', 'Longevity', 'Ultimate Impact'};
% 
% % 颜色定义（Top 20% 用红色，Bottom 20% 用蓝色）
% top_color = [238, 76, 76]/255;  % 红色
% bottom_color = [76, 136, 238]/255;  % 蓝色
% 
% % 创建子图
% figure;
% for i = 1:length(fields)
%     subplot(2, 4, i); % 创建2行4列子图
% 
%     % 提取当前字段的 Top 20% 和 Bottom 20% 数据
%     y_top = data_top.(fields{i});
%     y_bottom = data_bottom.(fields{i});
% 
%     % 绘制散点图（Top 20% - 红色空心）
%     scatter(x, y_top, 80, 'o', 'MarkerEdgeColor', top_color, ...
%         'MarkerFaceColor', 'none', 'LineWidth', 1.5);
%     hold on;
% 
%     % 绘制散点图（Bottom 20% - 蓝色空心）
%     scatter(x, y_bottom, 80, 'o', 'MarkerEdgeColor', bottom_color, ...
%         'MarkerFaceColor', 'none', 'LineWidth', 1.5);
% 
%     % **拟合曲线（Top 20% - 红色曲线）**
%     p_top = polyfit(x, y_top, 2); % 二次多项式拟合
%     x_fit = linspace(min(x), max(x), 100);
%     y_fit_top = polyval(p_top, x_fit);
%     plot(x_fit, y_fit_top, '-', 'Color', top_color, 'LineWidth', 2);
% 
%     % **拟合曲线（Bottom 20% - 蓝色曲线，虚线）**
%     p_bottom = polyfit(x, y_bottom, 2); % 二次多项式拟合
%     y_fit_bottom = polyval(p_bottom, x_fit);
%     plot(x_fit, y_fit_bottom, '--', 'Color', bottom_color, 'LineWidth', 2);
% 
%     % 设置坐标轴
%     xlabel('Rao-Stirling Index Bin', 'FontSize', 12);
%     ylabel(labels{i}, 'FontSize', 12);
%     set(gca, 'FontSize', 10);
%     xlim([0.5, length(x) + 0.5]); % 限制横轴范围
% 
%     % X 轴刻度匹配 `RS_bin`
%     xticks(1:3:length(x)); % 每隔 3 个 bin 显示一个标签
%     xticklabels(x_labels(1:3:end)); % 对应的 bin 标签
%     xtickangle(45); % 旋转刻度
% 
%     % 添加图例
%     if i == 1
%         legend({'Top 20% (Scatter)', 'Bottom 20% (Scatter)', ...
%                 'Top 20% (Fit)', 'Bottom 20% (Fit)'}, ...
%                 'Location', 'Best', 'FontSize', 10);
%     end
% end
% 
% % 调整布局
% set(gcf, 'Position', [100, 100, 1400, 700]); % 调整图形大小
