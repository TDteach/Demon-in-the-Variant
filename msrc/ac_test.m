home_folder = getenv('HOME');
fo = fullfile(home_folder,'/data/npys');
mat_folder = fullfile(home_folder,'/data/mats/backdoor');

fn = 'out';
[features,labels,ori_labels] = read_features(fn,fo);
[scores, gp, tpr, fpr, thr] = kmeans_defense(features, labels,ori_labels);

figure;
boxplot(scores(:,1), scores(:,2), 'PlotStyle','compact','symbol','.');

ylim([-0.5,1]);
set(gcf,'Position',[100 100 1000 200])
xlabel('label');
ylabel('silhouette score');
% hist(scores)
