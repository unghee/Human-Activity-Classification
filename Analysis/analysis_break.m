clear all

% CNN DATA ARE ALL HOP, BAND = 10

classifiers = ['SVM'; 'LDA';'CNN'];
% classifiers = ['SVM'; 'LDA';];
laterality=['bilateral'];
sensor = ['imu_emg_goin'];


for i = 1:size(classifiers,1)
    for j = 1:1
%         if strcmp(classifiers(i,:),'CNN')
%             filename = ['./' classifiers(i,:) '/' 'CNNbionet_bilateral_imu_emg_goin_BATCH_SIZE32_LR1e-05_WD0.001_EPOCH200_BAND10_HOP10_accuracy.txt'];
%         else
%             filename = ['./' classifiers(i,:) '/' classifiers(i,:) '_' laterality(j,:) '_' sensor(j,:) '_accuracy.txt'];
%         end
        filename = ['./' classifiers(i,:) '/' classifiers(i,:) '_' laterality(j,:) '_' sensor(j,:) '_accuracy.txt'];
        T=readtable(filename );
        data_temp = table2array(T(1:3,2:11));
        data.(classifiers(i,:)).(laterality(j,:)).(sensor(j,:)).overall= (1-data_temp(1,:))*100; 
        data.(classifiers(i,:)).(laterality(j,:)).(sensor(j,:)).steadystate= (1-data_temp(2,:))*100; 
        data.(classifiers(i,:)).(laterality(j,:)).(sensor(j,:)).transitional= (1-data_temp(3,:))*100;
 
        
    end
    
end    

% classifiers2 = ['CNN'];

% for i = 1:size(classifiers,1)
%     for j = 1:1
%         filename = ['./' classifiers(i,:) '/' 'CNNbionet_bilateral_imu_emg_goin_BATCH_SIZE32_LR1e-05_WD0.001_EPOCH200_BAND10_HOP10_accuracy.txt'];
%         T=readtable(filename );
%         data_temp = table2array(T(1:3,2:11));
%         data.(classifiers(i,:)).(laterality(j,:)).(sensor(j,:)).overall= (1-data_temp(1,:))*100; 
%         data.(classifiers(i,:)).(laterality(j,:)).(sensor(j,:)).steadystate= (1-data_temp(2,:))*100; 
%         data.(classifiers(i,:)).(laterality(j,:)).(sensor(j,:)).transitional= (1-data_temp(3,:))*100;   
%         
%     end
%     
% end 


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

% CNN_bilateral_overall_mean = (1-(mean(Freq_bilateral)))*100;
% CNN_bilateral_overall_std = std(Freq_bilateral)*100;

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


y_overall = [ y_base_overall; mean(LDA_bilateral.overall); mean(SVM_bilateral.overall); mean(CNN_bilateral.overall);]
stds_overall = [0; std(LDA_bilateral.overall) ; std(SVM_bilateral.overall); std(CNN_bilateral.overall)]

y_ss = [ y_base_ss; mean(LDA_bilateral.steadystate);  mean(SVM_bilateral.steadystate); mean(CNN_bilateral.steadystate)]
stds_ss = [0; std(LDA_bilateral.overall) ; std(SVM_bilateral.overall); std(CNN_bilateral.steadystate)]

y_tr = [ y_base_tr ;mean(LDA_bilateral.transitional);  mean(SVM_bilateral.transitional); mean(CNN_bilateral.transitional)]
stds_tr = [0; std(LDA_bilateral.transitional) ; std(SVM_bilateral.transitional); std(CNN_bilateral.transitional)]

y_overall_mode = [ y_base_overall_mode; mean(LDA_bilateral_mode.overall); mean(SVM_bilateral_mode.overall); mean(CNN_bilateral_mode.overall)]
stds_overall_mode  = [ 0;std(LDA_bilateral_mode.overall) ; std(SVM_bilateral_mode.overall); std(CNN_bilateral_mode.overall)]

y_ss_mode  = [  y_base_ss_mode; mean(LDA_bilateral_mode.steadystate);  mean(SVM_bilateral_mode.steadystate); mean(CNN_bilateral_mode.steadystate)]
stds_ss_mode  = [ 0;std(LDA_bilateral_mode.overall) ; std(SVM_bilateral_mode.overall); std(CNN_bilateral_mode.steadystate)]

y_tr_mode  = [ y_base_tr_mode; mean(LDA_bilateral_mode.transitional);  mean(SVM_bilateral_mode.transitional); mean(CNN_bilateral_mode.transitional) ]
stds_tr_mode  = [ 0;std(LDA_bilateral.transitional) ; std(SVM_bilateral_mode.transitional); std(CNN_bilateral_mode.transitional)]



%%


classfiers_cat = [CNN_bilateral.overall' LDA_bilateral.overall' SVM_bilateral.overall'];

[~,~,stats] = anova1(classfiers_cat);

[c,~,~,gnames] = multcompare(stats);
% [gnames(c(:,1)), gnames(c(:,1)), num2cell(c(:,3:6))]
%%
close all

line_width = 2;
ratio = 0.8;

color_1 = [ 0, 36.5, 36.5]/100;
color_2 = [0, 46.3, 46.3]/100;
color_3 = [ 14.1, 62.4, 62.4]/100;

NAME = {'Base', 'LDA', 'SVM', 'CNN'};


% ylimit=20;
%%%

%  set(gcf, 'Position',  [100, 100, 600, 800])

fh3=figure
sb1=subplot(3,2,1)

h=bar(diag(y_overall),ratio,'stacked','LineWidth',1.5);

h(1).FaceColor =color_1;
h(1).EdgeColor =color_1;
h(2).FaceColor =color_1;
h(2).EdgeColor =color_1;
h(3).FaceColor =color_1;
h(3).EdgeColor =color_1;
h(4).FaceColor =color_3;
h(4).EdgeColor =color_3;


hold on

er = errorbar(y_overall,stds_overall);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
er.LineWidth = 2.5;
ylabel('Overall Error (%)')

plot(xlim,[min(y_overall) min(y_overall)], '--','LineWidth',line_width,'Color',[0.75 0.75 0.75])



legend('off')
box off
names = NAME;
set(gca, 'FontSize', 20,'Fontname','Times','xticklabel',names) 
ylim([0 50])
xlim([0 5])
breakyaxis([8 45]);
% break_axis();
title('Non mode-specific')

ah1 = axes('Parent',fh3,'Units','normalized','Position',[0.1 0.1 25 0.8]);


%%%
subplot(3,2,3)

h2=bar(diag(y_ss),ratio,'stacked','LineWidth',1.5);
h2(1).FaceColor =color_1;
h2(1).EdgeColor =color_1;
h2(2).FaceColor =color_1;
h2(2).EdgeColor =color_1;
h2(3).FaceColor =color_3;
h2(3).EdgeColor =color_3;



hold on

er = errorbar(y_ss,stds_ss);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
er.LineWidth = 2.5;

plot(xlim,[min(y_ss) min(y_ss)], '--','LineWidth',line_width,'Color',[0.75 0.75 0.75])



ylabel('Steady state Error (%)')
legend('off')
box off
names = NAME;
set(gca, 'FontSize', 20,'Fontname','Times','xticklabel',names) 
ylim([0 50])
xlim([0 5])
breakyaxis([8 45]);

%%%
subplot(3,2,5)

h=bar(diag(y_tr),ratio,'stacked','LineWidth',1.5);
h(1).FaceColor =color_1;
h(1).EdgeColor =color_1;
h(2).FaceColor =color_2;
h(2).EdgeColor =color_2;
h(3).FaceColor =color_3;
h(3).EdgeColor =color_3;



hold on

er = errorbar(y_tr,stds_tr);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
er.LineWidth = 2.5;

ylabel('Transitional Error (%)')


plot(xlim,[min(y_tr) min(y_tr)], '--','LineWidth',line_width,'Color',[0.75 0.75 0.75])


legend('off')
box off
names = NAME;
set(gca, 'FontSize', 20,'Fontname','Times','xticklabel',names) 
ylim([0 100])
xlim([0 5])
breakyaxis([60 80]);


%%% mode-specific

%%%

subplot(3,2,2)
ratio = 0.8;
h=bar(diag(y_overall_mode),ratio,'stacked','FaceColor',[0 .5 .5],'EdgeColor',[0 .5 .5],'LineWidth',1.5);
h(1).FaceColor =color_1;
h(1).EdgeColor =color_1;
h(2).FaceColor =color_2;
h(2).EdgeColor =color_2;
h(3).FaceColor =color_3;
h(3).EdgeColor =color_3;



hold on

er = errorbar(y_overall_mode,stds_overall_mode);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
er.LineWidth = 2.5;
% ylabel('Overall Error (%)')
plot(xlim,[min(y_overall_mode) min(y_overall_mode)], '--','LineWidth',line_width,'Color',[0.75 0.75 0.75])

legend('off')
box off
names = NAME;
set(gca, 'FontSize', 20,'Fontname','Times','xticklabel',names) 
ylim([0 50])
xlim([0 5])
breakyaxis([8 15]);

title('Mode-specific')




%%%
subplot(3,2,4)

h=bar(diag(y_ss_mode),ratio,'stacked','LineWidth',1.5);
h(1).FaceColor =color_1;
h(1).EdgeColor =color_1;
h(2).FaceColor =color_2;
h(2).EdgeColor =color_2;
h(3).FaceColor =color_3;
h(3).EdgeColor =color_3;


hold on

er = errorbar(y_ss_mode,stds_ss_mode);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
er.LineWidth = 2.5;

plot(xlim,[min(y_ss_mode) min(y_ss_mode)], '--','LineWidth',line_width,'Color',[0.75 0.75 0.75])


% ylabel('Steady state Error (%)')
legend('off')
box off
names =NAME;
set(gca, 'FontSize', 20,'Fontname','Times','xticklabel',names) 
ylim([0 4])
%%%
subplot(3,2,6)

h=bar(diag(y_tr_mode),ratio,'stacked','LineWidth',1.5);
h(1).FaceColor =color_1;
h(1).EdgeColor =color_1;
h(2).FaceColor =color_2;
h(2).EdgeColor =color_2;
h(3).FaceColor =color_3;
h(3).EdgeColor =color_3;

hold on

er = errorbar(y_tr_mode,stds_tr_mode);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
er.LineWidth = 2.5;

% ylabel('Transitional Error (%)')

plot(xlim,[min(y_tr_mode) min(y_tr_mode)], '--','LineWidth',line_width,'Color',[0.75 0.75 0.75])

legend('off')
box off
names = NAME;
set(gca, 'FontSize', 20,'Fontname','Times','xticklabel',names) 
ylim([0 35])

