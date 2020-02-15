clear

clear

classifiers = {'CNN'};
laterality={'bilateral','contralateral','ipsilateral'};
sensor = {'imu','emg','goin','imu_emg','emg_goin','imu_goin','imu_emg_goin'};

for i = 1:size(classifiers,1)
    for j = 1:length(laterality)
        for m = 1:length(sensor)

        if strcmp(sensor{m},'imu_emg_goin') && strcmp(laterality{j},'bilateral')
            filename = ['./' classifiers{i} '/' classifiers{i} '_' laterality{j} '_' sensor{m} '_accuracy.txt'];
        else    
            filename = ['./sensor_modalities_result/10_hop_10_band/' classifiers{i} '_' laterality{j} '_' sensor{m} '_BANDS_10_HOP_10_accuracy.txt'];
        end
        T=readtable(filename );
        data_temp = table2array(T(1:3,2:11));
        data.(classifiers{i}).(laterality{j}).(sensor{m}).overall= (1-data_temp(1,:))*100; 
        data.(classifiers{i}).(laterality{j}).(sensor{m}).steadystate= (1-data_temp(2,:))*100; 
        data.(classifiers{i}).(laterality{j}).(sensor{m}).transitional= (1-data_temp(3,:))*100;
        
        end
        
    end
    
end 


% 1 sensor

%imu
[mean(data.CNN.ipsilateral.imu.overall), std(data.CNN.ipsilateral.imu.overall)]
[mean(data.CNN.contralateral.imu.overall), std(data.CNN.contralateral.imu.overall)]
[mean(data.CNN.bilateral.imu.overall), std(data.CNN.bilateral.imu.overall)]

%gonio
[mean(data.CNN.ipsilateral.goin.overall), std(data.CNN.ipsilateral.goin.overall)]
[mean(data.CNN.contralateral.goin.overall), std(data.CNN.contralateral.goin.overall)]
[mean(data.CNN.bilateral.goin.overall), std(data.CNN.bilateral.goin.overall)]


%EMG
[mean(data.CNN.ipsilateral.emg.overall), std(data.CNN.ipsilateral.emg.overall)]
[mean(data.CNN.contralateral.emg.overall), std(data.CNN.contralateral.emg.overall)]
[mean(data.CNN.bilateral.emg.overall), std(data.CNN.bilateral.emg.overall)]


% 2 sensors

%imu + gon
[mean(data.CNN.ipsilateral.imu_goin.overall), std(data.CNN.ipsilateral.imu_goin.overall)]
[mean(data.CNN.contralateral.imu_goin.overall), std(data.CNN.contralateral.imu_goin.overall)]
[mean(data.CNN.bilateral.imu_goin.overall), std(data.CNN.bilateral.imu_goin.overall)]


%imu + emg
[mean(data.CNN.ipsilateral.imu_emg.overall), std(data.CNN.ipsilateral.imu_emg.overall)]
[mean(data.CNN.contralateral.imu_emg.overall), std(data.CNN.contralateral.imu_emg.overall)]
[mean(data.CNN.bilateral.imu_emg.overall), std(data.CNN.bilateral.imu_emg.overall)]

%emg+ goin
[mean(data.CNN.ipsilateral.emg_goin.overall), std(data.CNN.ipsilateral.emg_goin.overall)]
[mean(data.CNN.contralateral.emg_goin.overall), std(data.CNN.contralateral.emg_goin.overall)]
[mean(data.CNN.bilateral.emg_goin.overall), std(data.CNN.bilateral.emg_goin.overall)]


% 3 SENSORS
[mean(data.CNN.ipsilateral.imu_emg_goin.overall), std(data.CNN.ipsilateral.imu_emg_goin.overall)]
[mean(data.CNN.contralateral.imu_emg_goin.overall), std(data.CNN.contralateral.imu_emg_goin.overall)]
[mean(data.CNN.bilateral.imu_emg_goin.overall), std(data.CNN.bilateral.imu_emg_goin.overall)]


%% ANOVA TABLE


a_tab = [ data.CNN.ipsilateral.imu.overall' data.CNN.contralateral.imu.overall' data.CNN.bilateral.imu.overall';
data.CNN.ipsilateral.goin.overall' data.CNN.contralateral.goin.overall' data.CNN.bilateral.goin.overall';    
data.CNN.ipsilateral.emg.overall' data.CNN.contralateral.emg.overall' data.CNN.bilateral.emg.overall'; 

data.CNN.ipsilateral.imu_goin.overall' data.CNN.contralateral.imu_goin.overall' data.CNN.bilateral.imu_goin.overall'; 
data.CNN.ipsilateral.imu_emg.overall' data.CNN.contralateral.imu_emg.overall' data.CNN.bilateral.imu_emg.overall'; 
data.CNN.ipsilateral.emg_goin.overall' data.CNN.contralateral.emg_goin.overall' data.CNN.bilateral.emg_goin.overall'; 

data.CNN.ipsilateral.imu_emg_goin.overall' data.CNN.contralateral.imu_emg_goin.overall' data.CNN.bilateral.emg_goin.overall';

];


[p,tbl,stats] = anova2(a_tab,10);
c_lateral = multcompare(stats)
csensors = multcompare(stats,'Estimate','row')