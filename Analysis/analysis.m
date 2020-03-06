clear all

% CNN DATA ARE ALL HOP, BAND = 10

classifiers = ['SVM'; 'LDA';'CNN'];
laterality=['bilateral'];
sensor = ['imu_emg_goin'];


for i = 1:size(classifiers,1)
    for j = 1:1
        filename = ['./' classifiers(i,:) '/' classifiers(i,:) '_' laterality(j,:) '_' sensor(j,:) '_accuracy.txt'];
        T=readtable(filename );
        data_temp = table2array(T(1:3,2:11));
        data.(classifiers(i,:)).(laterality(j,:)).(sensor(j,:)).overall= (1-data_temp(1,:))*100; 
        data.(classifiers(i,:)).(laterality(j,:)).(sensor(j,:)).steadystate= (1-data_temp(2,:))*100; 
        data.(classifiers(i,:)).(laterality(j,:)).(sensor(j,:)).transitional= (1-data_temp(3,:))*100;   
    end
    
end    



for i = 1:size(classifiers,1)
    filename = ['./Mode_specific/' classifiers(i,:) '_bilateral_imu_emg_goin_accuracy_nway.txt'];
    T=readtable(filename );
    data_temp = table2array(T(1:3,2:11));
    data.modespecific.(classifiers(i,:)).overall = (1-data_temp(1,:))*100;  
    data.modespecific.(classifiers(i,:)).steadystate = (1-data_temp(2,:))*100; 
    data.modespecific.(classifiers(i,:)).transitional = (1-data_temp(3,:))*100; 
    
end

% LDA_mode_mean = 1.43;
% LDA_mode_std = 0.24;

%%

CNN_bilateral=data.(classifiers(3,:)).(laterality(1,:)).(sensor(1,:));
LDA_bilateral=data.(classifiers(2,:)).(laterality(1,:)).(sensor(1,:));
SVM_bilateral=data.(classifiers(1,:)).(laterality(1,:)).(sensor(1,:));

CNN_bilateral_mode=data.modespecific.(classifiers(3,:));
LDA_bilateral_mode=data.modespecific.(classifiers(2,:));
SVM_bilateral_mode=data.modespecific.(classifiers(1,:));


y_base_overall=100-52.01;
y_base_ss=100-52.57;
y_base_tr = 100- 49.54;

y_base_overall_mode = 100-81.52;
y_base_ss_mode = 100-100;
y_base_tr_mode = 100-0;


y_overall = [  mean(LDA_bilateral.overall) mean(SVM_bilateral.overall) mean(CNN_bilateral.overall);...
    mean(LDA_bilateral_mode.overall) mean(SVM_bilateral_mode.overall) mean(CNN_bilateral_mode.overall)]
stds_overall = [ std(LDA_bilateral.overall)  std(SVM_bilateral.overall) std(CNN_bilateral.overall);...
    std(LDA_bilateral_mode.overall)  std(SVM_bilateral_mode.overall) std(CNN_bilateral_mode.overall)]

y_ss = [  mean(LDA_bilateral.steadystate)  mean(SVM_bilateral.steadystate) mean(CNN_bilateral.steadystate);...
    mean(LDA_bilateral_mode.steadystate)  mean(SVM_bilateral_mode.steadystate) mean(CNN_bilateral_mode.steadystate)]
stds_ss = [ std(LDA_bilateral.steadystate)  std(SVM_bilateral.steadystate) std(CNN_bilateral.steadystate);...
 std(LDA_bilateral_mode.steadystate)  std(SVM_bilateral_mode.steadystate) std(CNN_bilateral_mode.steadystate)]

y_tr = [ mean(LDA_bilateral.transitional)  mean(SVM_bilateral.transitional) mean(CNN_bilateral.transitional);...
    mean(LDA_bilateral_mode.transitional)  mean(SVM_bilateral_mode.transitional) mean(CNN_bilateral_mode.transitional)]
stds_tr = [ std(LDA_bilateral.transitional)  std(SVM_bilateral.transitional) std(CNN_bilateral.transitional);...
    std(LDA_bilateral_mode.transitional)  std(SVM_bilateral_mode.transitional) std(CNN_bilateral_mode.transitional)]


%%


classfiers_overall = [ LDA_bilateral.overall' SVM_bilateral.overall' CNN_bilateral.overall';
   LDA_bilateral_mode.overall' SVM_bilateral_mode.overall' CNN_bilateral_mode.overall' ];
classfiers_steadystate = [ LDA_bilateral.steadystate' SVM_bilateral.steadystate' CNN_bilateral.steadystate';
    LDA_bilateral_mode.steadystate' SVM_bilateral_mode.steadystate' CNN_bilateral_mode.steadystate'];
classfiers_transitional = [ LDA_bilateral.transitional' SVM_bilateral.transitional' CNN_bilateral.transitional';
    LDA_bilateral_mode.transitional' SVM_bilateral_mode.transitional' CNN_bilateral_mode.transitional'];


[~,~,stats_overall] = anova2(classfiers_overall,10);
[c,~,~,gnames] = multcompare(stats_overall)
c_config = multcompare(stats_overall,'Estimate','row')

[~,~,stats_steadystate] = anova2(classfiers_steadystate,10);
[c,~,~,gnames] = multcompare(stats_steadystate);
c_config_ss = multcompare(stats_steadystate,'Estimate','row')

[~,~,stats_transitional] = anova2(classfiers_transitional ,10);
[c,~,~,gnames] = multcompare(stats_transitional);
c_config_tr = multcompare(stats_transitional,'Estimate','row')


% [c,~,~,gnames] = multcompare(stats,'CType','bonferroni') % larger
% confidence interval



%%
close all

line_width = 2;
% ratio = 0.8;

color_1 = [ 0, 36.5, 36.5]/100;
color_2 = [0, 46.3, 46.3]/100;
color_3 = [ 14.1, 62.4, 62.4]/100;

NAME = {'Generic', 'Mode-specific'};


fh3=figure

set(gcf, 'Position',  [100, 100, 600, 1000])
subplot(3,1,1)

h=bar(y_overall ,'LineWidth',1.5,'BarWidth', 0.95);


h(1).FaceColor =color_1;
h(1).EdgeColor =color_1;
h(2).FaceColor =color_2;
h(2).EdgeColor =color_2;
h(3).FaceColor =color_3;
h(3).EdgeColor =color_3;


hold on


ngroups = size(y_overall, 1);
nbars = size(y_overall, 2);
% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
% Set the position of each error bar in the centre of the main bar
% Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, y_overall(:,i), stds_overall(:,i), 'k', 'linestyle', 'none','linewidth',2.5);
end




er.Color = [0 0 0];                            
er.LineStyle = 'none';  
er.LineWidth = 2.5;
ylabel('Overall Error (%)')

legend({'LDA','SVM','LAP-NET'},'Box','off')
box off
names = NAME;
set(gca, 'FontSize', 20,'Fontname','Times','xticklabel',names) 
ylim([0 8])
% % xlim([0 4])

% plot(xlim,[min(y_overall,'all') min(y_overall,'all')], '--','LineWidth',line_width,'Color',[0.75 0.75 0.75])

%%%
subplot(3,1,2)

h2=bar(y_ss,'LineWidth',1.5,'BarWidth', 0.95);

h2(1).FaceColor =color_1;
h2(1).EdgeColor =color_1;
h2(2).FaceColor =color_2;
h2(2).EdgeColor =color_2;
h2(3).FaceColor =color_3;
h2(3).EdgeColor =color_3;

hold on

% er = errorbar(y_ss,stds_ss);   
for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, y_ss(:,i), stds_ss(:,i), 'k', 'linestyle', 'none','linewidth',2.5);
end

ylabel('Steady state Error (%)')

box off
names = NAME;
set(gca, 'FontSize', 20,'Fontname','Times','xticklabel',names) 

%%%
subplot(3,1,3)

h=bar(y_tr,'LineWidth',1.5,'BarWidth', 0.95);

h(1).FaceColor =color_1;
h(1).EdgeColor =color_1;
h(2).FaceColor =color_2;
h(2).EdgeColor =color_2;
h(3).FaceColor =color_3;
h(3).EdgeColor =color_3;

hold on

for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, y_tr(:,i), stds_tr(:,i), 'k', 'linestyle', 'none','linewidth',2.5);
end


ylabel('Transitional Error (%)')

legend('off')
box off
names = NAME;
set(gca, 'FontSize', 20,'Fontname','Times','xticklabel',names) 