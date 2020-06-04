home_folder = getenv('HOME');
fo = fullfile(home_folder,'/data/npys');
mat_folder = fullfile(home_folder,'/data/mats/backdoor');

fn = 'out';
[features,labels,ori_labels] = read_features(fn,fo);


[scores, s0, v0] = ss_defense(features, labels,ori_labels);
% 
% figure;
% hist(scores);
