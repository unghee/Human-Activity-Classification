clear all

% CNN DATA ARE ALL HOP, BAND = 10 LR1e-05_WD0.001_EPOCH200_BAND10_HOP10 BATCH_SIZE32_

classifiers = {'Random','SVM', 'LDA','CNN'};
laterality=['bilateral'];
sensor = ['imu_emg_goin'];


for i = 1:size(classifiers,2)
    for j = 1:1
        filename = ['./' classifiers{i} '/' classifiers{i} '_' laterality(j,:) '_' sensor(j,:) '_accuracy.txt'];
        T=readtable(filename );
        data_temp = table2array(T(1:3,2:11));
        data.(classifiers{i}).(laterality(j,:)).(sensor(j,:)).overall= (1-data_temp(1,:))*100; 
        data.(classifiers{i}).(laterality(j,:)).(sensor(j,:)).steadystate= (1-data_temp(2,:))*100; 
        data.(classifiers{i}).(laterality(j,:)).(sensor(j,:)).transitional= (1-data_temp(3,:))*100;   
    end
    
end    



for i = 1:size(classifiers,2)
    filename = ['./Mode_specific/' classifiers{i} '_bilateral_imu_emg_goin_accuracy_nway.txt'];
    T=readtable(filename );
    data_temp = table2array(T(1:3,2:11));
    data.modespecific.(classifiers{i}).overall = (1-data_temp(1,:))*100;  
    data.modespecific.(classifiers{i}).steadystate = (1-data_temp(2,:))*100; 
    data.modespecific.(classifiers{i}).transitional = (1-data_temp(3,:))*100; 
    
end

% independent 

for i = 1:size(classifiers,2)
    filename = ['./independent/' classifiers{i} '_bilateral_imu_emg_goin_subjects_accuracy.txt'];
    T=readtable(filename );
    data_temp = table2array(T(1:3,2:11));
    data.indi.(classifiers{i}).overall = (1-data_temp(1,:))*100;  
    data.indi.(classifiers{i}).steadystate = (1-data_temp(2,:))*100; 
    data.indi.(classifiers{i}).transitional = (1-data_temp(3,:))*100; 
    
end


for i = 1:size(classifiers,2)
    filename = ['./independent/' classifiers{i} '_bilateral_imu_emg_goin_accuracy_nway_subject.txt'];
    T=readtable(filename );
    data_temp = table2array(T(1:3,2:11));
    data.indi.modespecific.(classifiers{i}).overall = (1-data_temp(1,:))*100;  
    data.indi.modespecific.(classifiers{i}).steadystate = (1-data_temp(2,:))*100; 
    data.indi.modespecific.(classifiers{i}).transitional = (1-data_temp(3,:))*100; 
    
end


%% 

CNN_bilateral=data.(classifiers{4}).(laterality(1,:)).(sensor(1,:));
LDA_bilateral=data.(classifiers{3}).(laterality(1,:)).(sensor(1,:));
SVM_bilateral=data.(classifiers{2}).(laterality(1,:)).(sensor(1,:));
RAN_bilateral=data.(classifiers{1}).(laterality(1,:)).(sensor(1,:));


CNN_bilateral_mode=data.modespecific.(classifiers{4});
LDA_bilateral_mode=data.modespecific.(classifiers{3});
SVM_bilateral_mode=data.modespecific.(classifiers{2});
RAN_bilateral_mode=data.modespecific.(classifiers{1});


CNN_bilateral_indi=data.indi.(classifiers{4});
LDA_bilateral_indi=data.indi.(classifiers{3});
SVM_bilateral_indi=data.indi.(classifiers{2});
RAN_bilateral_indi=data.indi.(classifiers{1});


CNN_bilateral_mode_indi=data.indi.modespecific.(classifiers{4});
LDA_bilateral_mode_indi=data.indi.modespecific.(classifiers{3});
SVM_bilateral_mode_indi=data.indi.modespecific.(classifiers{2});
RAN_bilateral_mode_indi=data.indi.modespecific.(classifiers{1});


%%

classfiers_overall = [ RAN_bilateral.overall LDA_bilateral.overall SVM_bilateral.overall CNN_bilateral.overall...
   RAN_bilateral_mode.overall LDA_bilateral_mode.overall SVM_bilateral_mode.overall CNN_bilateral_mode.overall...
   RAN_bilateral_indi.overall LDA_bilateral_indi.overall SVM_bilateral_indi.overall CNN_bilateral_indi.overall...
   RAN_bilateral_mode_indi.overall LDA_bilateral_mode_indi.overall SVM_bilateral_mode_indi.overall CNN_bilateral_mode_indi.overall
   ];

classfiers_ss= [ RAN_bilateral.steadystate LDA_bilateral.steadystate SVM_bilateral.steadystate CNN_bilateral.steadystate...
    RAN_bilateral_mode.steadystate LDA_bilateral_mode.steadystate SVM_bilateral_mode.steadystate CNN_bilateral_mode.steadystate...
    RAN_bilateral_indi.steadystate LDA_bilateral_indi.steadystate SVM_bilateral_indi.steadystate CNN_bilateral_indi.steadystate...
    RAN_bilateral_mode_indi.steadystate LDA_bilateral_mode_indi.steadystate SVM_bilateral_mode_indi.steadystate CNN_bilateral_mode_indi.steadystate
       
    
    ];

classfiers_tr = [ RAN_bilateral.transitional LDA_bilateral.transitional SVM_bilateral.transitional CNN_bilateral.transitional...
    RAN_bilateral_mode.transitional LDA_bilateral_mode.transitional SVM_bilateral_mode.transitional CNN_bilateral_mode.transitional...
    RAN_bilateral_indi.transitional LDA_bilateral_indi.transitional SVM_bilateral_indi.transitional CNN_bilateral_indi.transitional...
    RAN_bilateral_mode_indi.transitional LDA_bilateral_mode_indi.transitional SVM_bilateral_mode_indi.transitional CNN_bilateral_mode_indi.transitional];
       


% g1 = {'RAN'; 'LDA'; 'SVM' ; 'CNN';...
%     'RAN'; 'LDA'; 'SVM' ; 'CNN';...
%     'RAN'; 'LDA'; 'SVM' ; 'CNN';...
% %     'RAN'; 'LDA'; 'SVM' ; 'CNN'}; 
% g2 = {'gen';'gen';'gen';'gen';...
%     'mode';'mode';'mode';'mode';...
%     'gen';'gen';'gen';'gen';...
%     'mode';'mode';'mode';'mode';...
%     }; 
% g3 = {'dep';'dep';'dep';'dep';...
%     'dep';'dep';'dep';'dep';...
%     'indi';'indi';'indi';'indi';...
%     'indi';'indi';'indi';'indi'};

g1 = [repmat('RAN',10,1); repmat('LDA',10,1); repmat('SVM',10,1) ; repmat('CNN',10,1);...
    repmat('RAN',10,1); repmat('LDA',10,1); repmat('SVM',10,1) ; repmat('CNN',10,1);...
    repmat('RAN',10,1); repmat('LDA',10,1); repmat('SVM',10,1) ; repmat('CNN',10,1);...
    repmat('RAN',10,1); repmat('LDA',10,1); repmat('SVM',10,1) ; repmat('CNN',10,1);...
    ];

g2 = [repmat({'gen'},10,1); repmat({'gen'},10,1); repmat({'gen'},10,1) ; repmat({'gen'},10,1);...
    repmat({'mod'},10,1); repmat({'mod'},10,1); repmat({'mod'},10,1) ; repmat({'mod'},10,1);...
    repmat({'gen'},10,1); repmat({'gen'},10,1); repmat({'gen'},10,1) ; repmat({'gen'},10,1);...
    repmat({'mod'},10,1); repmat({'mod'},10,1); repmat({'mod'},10,1) ; repmat({'mod'},10,1);...
    ];

g3 = [repmat({'dep'},10,1); repmat({'dep'},10,1); repmat({'dep'},10,1) ; repmat({'dep'},10,1);...
    repmat({'dep'},10,1); repmat({'dep'},10,1); repmat({'dep'},10,1) ; repmat({'dep'},10,1);...
    repmat({'ind'},10,1); repmat({'ind'},10,1); repmat({'ind'},10,1) ; repmat({'ind'},10,1);...
    repmat({'ind'},10,1); repmat({'ind'},10,1); repmat({'ind'},10,1) ; repmat({'ind'},10,1);...
    ];




%% ANOVA

[~,~,stats] = anovan(classfiers_overall,{g1,g2,g3},'model','interaction',...
    'varnames',{'classifer','mode','dependency'});

[~,~,stats_ss] = anovan(classfiers_ss,{g1,g2,g3},'model','interaction',...
    'varnames',{'classifer','mode','dependency'});

[~,~,stats_tr] = anovan(classfiers_tr,{g1,g2,g3},'model','interaction',...
    'varnames',{'classifer','mode','dependency'});


% effect of classifer
results = multcompare(stats,'Dimension',[1],'Display','off')
results_ss = multcompare(stats_ss,'Dimension',[1],'Display','off')
results_tr = multcompare(stats_tr,'Dimension',[1],'Display','off')

% effect of mode
multcompare(stats,'Dimension',[2],'Display','off')
multcompare(stats_ss,'Dimension',[2],'Display','off')
multcompare(stats_tr,'Dimension',[2],'Display','off')


% effect of dependency
multcompare(stats,'Dimension',[3],'Display','off')
multcompare(stats_ss,'Dimension',[3],'Display','off')
multcompare(stats_tr,'Dimension',[3],'Display','off')

%%
% ANOVA2
% classfiers_overall = [ RAN_bilateral.overall' LDA_bilateral.overall' SVM_bilateral.overall' CNN_bilateral.overall';
%    RAN_bilateral_mode.overall' LDA_bilateral_mode.overall' SVM_bilateral_mode.overall' CNN_bilateral_mode.overall' ];
% classfiers_steadystate = [ RAN_bilateral.steadystate' LDA_bilateral.steadystate' SVM_bilateral.steadystate' CNN_bilateral.steadystate';
%     RAN_bilateral_mode.steadystate' LDA_bilateral_mode.steadystate' SVM_bilateral_mode.steadystate' CNN_bilateral_mode.steadystate'];
% classfiers_transitional = [ RAN_bilateral.transitional' LDA_bilateral.transitional' SVM_bilateral.transitional' CNN_bilateral.transitional';
%     RAN_bilateral_mode.transitional' LDA_bilateral_mode.transitional' SVM_bilateral_mode.transitional' CNN_bilateral_mode.transitional'];
% 
% 
% [~,~,stats_overall] = anova2(classfiers_overall,10);
% [c,~,~,gnames] = multcompare(stats_overall)
% c_config = multcompare(stats_overall,'Estimate','row')
% 
% [~,~,stats_steadystate] = anova2(classfiers_steadystate,10);
% [c_ss,~,~,gnames] = multcompare(stats_steadystate)
% c_config_ss = multcompare(stats_steadystate,'Estimate','row')
% 
% [~,~,stats_transitional] = anova2(classfiers_transitional ,10);
% [c_tr,~,~,gnames] = multcompare(stats_transitional)
% c_config_tr = multcompare(stats_transitional,'Estimate','row')
% 

% ANOVA1
% [~,~,stats_overall] = anova1(classfiers_overall(1:10,1:end));
% [c,~,~,gnames] = multcompare(stats_overall)
% 
% [~,~,stats_steadystate] = anova1(classfiers_steadystate(1:10,1:end));
% [c,~,~,gnames] = multcompare(stats_steadystate)
% 
% 
% [~,~,stats_transitional] = anova1(classfiers_transitional(1:10,1:end));
% [c,~,~,gnames] = multcompare(stats_transitional)

% [c,~,~,gnames] = multcompare(stats,'CType','bonferroni') % larger
% confidence interval



%% FOR PLOTTING

y_overall = [  mean(LDA_bilateral.overall) mean(SVM_bilateral.overall) mean(CNN_bilateral.overall);...
    mean(LDA_bilateral_mode.overall) mean(SVM_bilateral_mode.overall) mean(CNN_bilateral_mode.overall);...
    mean(LDA_bilateral_indi.overall) mean(SVM_bilateral_indi.overall) mean(CNN_bilateral_indi.overall);...
    mean(LDA_bilateral_mode_indi.overall) mean(SVM_bilateral_mode_indi.overall) mean(CNN_bilateral_mode_indi.overall)
    ]


stds_overall = [ std(LDA_bilateral.overall)  std(SVM_bilateral.overall) std(CNN_bilateral.overall);...
    std(LDA_bilateral_mode.overall)  std(SVM_bilateral_mode.overall) std(CNN_bilateral_mode.overall);...
    std(LDA_bilateral_indi.overall) std(SVM_bilateral_indi.overall) std(CNN_bilateral_indi.overall);...
    std(LDA_bilateral_mode_indi.overall) std(SVM_bilateral_mode_indi.overall) std(CNN_bilateral_mode_indi.overall)
    ]

y_ss = [  mean(LDA_bilateral.steadystate)  mean(SVM_bilateral.steadystate) mean(CNN_bilateral.steadystate);
    mean(LDA_bilateral_mode.steadystate)  mean(SVM_bilateral_mode.steadystate) mean(CNN_bilateral_mode.steadystate);
    mean(LDA_bilateral_indi.steadystate)  mean(SVM_bilateral_indi.steadystate) mean(CNN_bilateral_indi.steadystate);
    mean(LDA_bilateral_mode_indi.steadystate)  mean(SVM_bilateral_mode_indi.steadystate) mean(CNN_bilateral_mode_indi.steadystate)
    ]



stds_ss = [ std(LDA_bilateral.steadystate)  std(SVM_bilateral.steadystate) std(CNN_bilateral.steadystate);...
 std(LDA_bilateral_mode.steadystate)  std(SVM_bilateral_mode.steadystate) std(CNN_bilateral_mode.steadystate);...
 std(LDA_bilateral_indi.steadystate)  std(SVM_bilateral_indi.steadystate) std(CNN_bilateral_indi.steadystate);...
 std(LDA_bilateral_mode_indi.steadystate)  std(SVM_bilateral_mode_indi.steadystate) std(CNN_bilateral_mode_indi.steadystate)
   
 ]

y_tr = [ mean(LDA_bilateral.transitional)  mean(SVM_bilateral.transitional) mean(CNN_bilateral.transitional);...
    mean(LDA_bilateral_mode.transitional)  mean(SVM_bilateral_mode.transitional) mean(CNN_bilateral_mode.transitional);...
    mean(LDA_bilateral_indi.transitional)  mean(SVM_bilateral_indi.transitional) mean(CNN_bilateral_indi.transitional);...
    mean(LDA_bilateral_mode_indi.transitional)  mean(SVM_bilateral_mode_indi.transitional) mean(CNN_bilateral_mode_indi.transitional)
   ]

stds_tr = [ std(LDA_bilateral.transitional)  std(SVM_bilateral.transitional) std(CNN_bilateral.transitional);...
    std(LDA_bilateral_mode.transitional)  std(SVM_bilateral_mode.transitional) std(CNN_bilateral_mode.transitional);
    std(LDA_bilateral_indi.transitional)  std(SVM_bilateral_indi.transitional) std(CNN_bilateral_indi.transitional);...
    std(LDA_bilateral_mode_indi.transitional)  std(SVM_bilateral_mode_indi.transitional) std(CNN_bilateral_mode_indi.transitional)
    ]
   

%%
close all

line_width = 2;
% ratio = 0.8;

color_1 = [ 0, 36.5, 36.5]/100;
color_2 = [0, 46.3, 46.3]/100;
color_3 = [ 14.1, 62.4, 62.4]/100;

NAME = {'Generic\newline{Dep.}', 'Mode\newline{Dep.}','Generic\newline{Indep.}', 'Mode\newline{Indep.}';
};


fh3=figure

set(gcf, 'Position',  [100, 100, 700, 1000])
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

legend({'LDA','SVM','LIR-NET'},'Box','off','Location','northwest')
box off
names = NAME;
set(gca, 'FontSize', 20,'Fontname','Times','xticklabel',names) 
ylim([0 25])
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

ylim([0 25])
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
ylim([0 45])

legend('off')
box off
names = NAME;
set(gca, 'FontSize', 20,'Fontname','Times','xticklabel',names) 
