%% data
Freq_bilateral=[0.9862745098039216
0.9877450980392157
0.9872549019607844
0.9843060323688082
0.9852869053457577
0.9872486512996567
0.9833251593918587
0.9906817067189799
0.9833251593918587
0.9877390877881315];


LDA_bilateral=[0.9356388088376562
0.9414024975984631
0.9442843419788665
0.9433237271853987
0.93611911623439
0.9390009606147934
0.9370797310278578
0.9370797310278578
0.9317963496637848
0.9380403458213257];




%%

Freq_bilateral_mean = (1-(mean(Freq_bilateral)))*100;
Freq_bilateral_std = std(Freq_bilateral)*100;

LDA_bilateral_mean = (1-(mean(LDA_bilateral)))*100;
LDA_bilateral_std = std(Freq_bilateral)*100;

LDA_mode_mean = 1.43;
LDA_mode_std = 0.24;


y = [ LDA_bilateral_mean ; LDA_mode_mean; Freq_bilateral_mean];
stds = [ LDA_bilateral_std ; LDA_mode_std; Freq_bilateral_std];

color_matrix_pos=[255 222 164 ;255 183 57; 255 144 0 ]/255; %shiny
color_matrix_cur=[128 158 255; 29 82 255; 0 34 147]/255; %purpleish_2

b=bar(y);
% b(1).FaceColor = color_matrix_cur(3,:); 
% b(2).FaceColor = color_matrix_cur(1,:) ;
% b(1).EdgeColor = color_matrix_cur(3,:); 
% b(2).EdgeColor = color_matrix_cur(1,:) ;

hold on

er = errorbar(y,stds);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
er.LineWidth = 2;


% legend('boxoff')
legend('off')

names = {'LDA', 'LDA-mode', 'CNN'};
ylabel('Error rate (%)')
box off
set(gca, 'FontSize', 20,'Fontname','Helvetica','xticklabel',names) 
ylim([0 7])
