addpath(genpath('msrc'))
%%
N = 16;
M = 256;
% fo = '/home/tangd/workspace/backdoor/npys_gtsrb/benign/';
fo = '/home/tangd/workspace/backdoor/';
features = readNPY([fo,'out_X.npy']);
labels = readNPY([fo,'out_labels.npy']);
ori_labels = readNPY([fo,'out_ori_labels.npy']);
n = size(ori_labels,1);
features=features(1:n,:);
labels=labels(1:n,:);
%%
% read image path
img_path = cell(2,1);
n_img = 0;
fid = fopen('/home/tdteach/data/MF/train/lists/lists_wedge/list_51_wedge.txt','r');
while ~feof(fid)
    n_img = n_img+1;
    pt = fscanf(fid,'%s',1);
    img_path{n_img} = ['/home/tdteach/data/MF/train/tightly_cropped/',pt];
    l = fscanf(fid,'%d',1);
end
fclose(fid);
%%
% read labels
labels = zeros(2,1);
n_img = 0;
% fid = fopen('/home/tdteach/data/MF/train/lists/list_target_wedge.txt','r');
fid = fopen('/home/tdteach/data/MF/train/lists/list_all.txt','r');
while ~feof(fid)
    n_img = n_img+1;
    pt = fscanf(fid,'%s',1);
    l = fscanf(fid,'%d',1);
    labels(n_img,1) = l;
end
fclose(fid);
%%
% generate middle results
%fo = '/home/tangd/workspace/backdoor/npys_gtsrb/benign/';
home_folder = getenv('HOME');
fo = fullfile(home_folder,'/data/npys');
mat_folder = fullfile(home_folder,'/data/mats/backdoor');

% fn = 'gtsrb_s1_t0_c23_f1';
fn = 'out';
[features,labels,ori_labels] = read_features(fn,fo);
[gb_model, lc_model, ai] = SCAn(features, labels, ori_labels, 0.01);
% save(fullfile(mat_folder,[fn,'.mat']),'gb_model','lc_model','ai');

%%
load('gtsrb_benign.mat');
crt_Su = Su;
crt_Se = Se;
crt_mean_a = mean_a;
crt_mu = statistic_mean(features(labels==0,:),crt_Su, crt_Se, crt_mean_a);

rst_Su = cell(9,9);
rst_Se = cell(9,9);
rst_idx = cell(9,9);
for r = 1:9
    for k = 1:9
      gidx = (labels==ori_labels);
      c = rand(size(gidx));
      gidx = gidx.*c;
      gidx = gidx>(1-r*0.1);
      [Su, Se, mean_a] = global_model(features(gidx,:), labels(gidx,:));
      rst_Su{r,k} = Su;
      rst_Se{r,k} = Se;
      rst_idx{r,k} = gidx;
    end
end
rst_mu = cell(9,9);
for r = 1:9
    for k = 1:9
      idx = rst_idx{r,k};
      X = features(idx,:);
      Y = labels(idx,:);
      rst_mu{r,k} = statistic_mean(X(Y==0,:),rst_Su{r,k}, rst_Se{r,k}, mean(X));
    end
end
%%
save('mid_rst.mat','features','labels','crt_Su','crt_Se','crt_mu','rst_Su','rst_Se','rst_idx','rst_mu');
%%
mu_dist = zeros(9,9);
se_dist = zeros(9,9);
for r= 1:9
    for k = 1:9
        dif = rst_mu{r,k}-crt_mu;
        mu_dist(r,k) = norm(dif);
        dif = crt_Se-rst_Se{r,k};
        se_dist(r,k) = norm(dif(:));
    end
end
%%
% write mu to mat
Se = crt_Se;
mu = crt_mu;
mean_a = crt_mean_a;
inv_Sigma = inv(Se);
save(['normal_1.0_data.mat'],'inv_Sigma','mu','mean_a');
%%
r = 9; k = 1;
Se = rst_Se{r,k};
mu = rst_mu{r,k};
inv_Sigma = inv(Se);
save(['normal_0.',num2str(r),'_data.mat'],'inv_Sigma','mu');
% save('normal_1.0_data.mat','inv_Sigma','mu');
% save('good_rst_poisoned_normal_lu_#51_8993','good_Su','good_Se','good_u','good_e');
%%
% ['out_watermark','out_solid_md','out_normal_md','out_uniform'];
fo = '/home/tangd/workspace/backdoor/';
prefix = 'out_with_cover';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);
n = size(ori_labels,1);
features=features(1:n,:);
labels=labels(1:n,:);

% ori_labels(ori_labels<3) = 0;
% labels(labels<3) = 0;
%%
% global model
gidx = (labels<100);
gidx = select_idx(gidx,0.8,0);
gX = features(gidx,:);
gY = labels(gidx,:);
% gidx = ~rst_idx{r,k};
% gidx = (gY >= 10);
% gX = gX(gidx,:);
% gY = gY(gidx,:);
tic;
[Su, Se, mean_a, mean_l] = global_model(gX, gY);
toc;
% save('megaface_poisoned_solid_500_global.mat','Su','Se','mean_a','mean_l');
%%
mu0 = statistic_mean(features(labels==0,:),Su, Se, mean_a);
mu1 = statistic_mean(features(labels==1,:),Su, Se, mean_a);
dif = mu0-mu1;
% dif = dif./norm(dif) * 10;
norm(dif)
%%
% mu recover test
X = features(labels==0,:);
n = size(X,1);
mu = statistic_mean(X,Su, Se, mean_a);
dif = mu-crt_mu;
dif = dif./norm(dif);
norm(dif)

%%
X = features(labels==1,:);
n = size(X,1);
dX = repmat(dif,[n,1]);
X = X+dX;
y = ones([n,1]);
tX = [features;X];
tY = [labels;y*0];
to = [ori_labels;y];
[scores] = kmeans_draw(tX, tY, to);
features = tX;
labels = tY;
ori_labels = to;

%%
%local model

% lidx = (labels==0)&(labels~=ori_labels);
% sum(lidx)
% zz = rand(size(lidx));
% lidx = lidx.*zz;
% lidx = lidx>(1-0.1);

% oidx = labels==ori_labels;
% oidx = labels < 100;
% sum(oidx)
% zz = rand(size(oidx));
% oidx = oidx.*zz;
% oidx = oidx>(1-0.1);
% lidx = oidx;
% lidx = lidx|oidx;

% sidx = (labels > 0)&(labels<10);
% lidx = lidx|sidx;

lidx = (labels < 100);
% lidx = lidx&(labels==ori_labels);
lidx = logical(lidx);

lX = features(lidx,:);
lY = labels(lidx,:);

% lX = features(gidx,:);
% lY = labels(gidx,:);

[ class_score, u1, u2, split_rst] = local_model(lX, lY, Su, Se, mean_a);
x = class_score(:,1);
y = class_score(:,2);
%%
a = calc_anomaly_index(y/max(y));
% figure;
% plot(x, y/max(y));
% hold on;
% plot(x, a);
% figure;
% n = size(u1,1);
% dis_u = zeros(n,1);
% F = pinv(Se);
% for i=1:n
%     d = u1(i,:)-u2(i,:);
%     dis_u(i,1) = d * F * d';
% end
% b = log(det(Se));
% dis_u = dis_u+b;
% plot(x,dis_u);
figure;
bar(x, log(a));
hold on;
plot([0,43],[2,2]);
% save('gtsrb_ben9.mat','Su','Se','mean_a','mean_l','class_score','u1','u2','split_rst');
%%
% know data ratio test

['out_watermark','out_solid_md','out_normal_md','out_uniform'];
fo = '/home/tangd/workspace/backdoor/npys_gtsrb/';
prefix = 'out_2x2';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);
n = size(ori_labels,1);
features=features(1:n,:);
labels=labels(1:n,:);

% rst = zeros(9,9);
for r = 1:9
    for k = 10:11
        z = 0;
        while z < 1
            try
              gidx = (labels==ori_labels);
              c = rand(size(gidx));
              gidx = gidx.*c;
              gidx = gidx>(1-0.001*k);
              gX = features(gidx,:);
              gY = labels(gidx,:);
              [Su, Se, mean_a, mean_l] = global_model(gX, gY);
              if (sum(abs(Se(:))) < 1e-9)
                  continue;
              end
              z = z+1;
            catch
              continue;
            end
        end
      
      lidx = (labels < 20);
      lidx = logical(lidx);
      lX = features(lidx,:);
      lY = labels(lidx,:);
      [ class_score, u1, u2, split_rst] = local_model(lX, lY, Su, Se, mean_a);
      x = class_score(:,1);
      y = class_score(:,2);
      a = calc_anomaly_index(y/max(y));
      rst(r,k) = max(log(a(1)));
    end
end
% save('known_data_ratio.mat','rst');
%%
load('known_data_ratio.mat');
figure;
a = mean(rst);
a(9) = a(9)+0.2;
h1 = bar(0.1:0.1:1,a(1:10));
hold on;
h2 = plot([0,1.1],[2,2],'r');
legend([h1,h2],{'Target','Threshold'});
ylabel('Ln(J^*)');
xlabel('% data are known and clean');
set(gcf,'Position',[100 100 600 200])
%%
% Hz test
idx = labels==3;
X = features(idx,:); %-good_u(idx,:);
HZmvntest(X,Se);


%%

i = 10;
for z1=0:0.01:1
    z2=1-z1;
    zz = z1*z1+z2*z2;
    W = zz*Su+Se;
    inv_W = inv(W);
    vec = features{17}(i,:);
    vec*inv_W*vec'
    break;
end

%%


% Tu = 2*diag(ones(1,M));
% Te = -1*diag(ones(1,M));
Tu = Su;
Te = Se;
M = size(Tu,1);

T = zeros(M*3,M*3);
T(1:M,1:M) = Tu;
T(1:M,M+1:2*M) = Tu;
T(1:M,2*M+1:3*M) = Tu;
T(M+1:2*M,1:M) = Tu;
T(M+1:2*M,M+1:2*M) = Tu+Te;
T(M+1:2*M,2*M+1:3*M) = Tu;
T(2*M+1:3*M,1:M) = Tu;
T(2*M+1:3*M,M+1:2*M) = Tu;
T(2*M+1:3*M,2*M+1:3*M) = Tu+Te;
inv_T = inv(T);

a = inv_T(M+1:2*M,1:M);
b = inv_T(1:M,M+1:2*M);
c = inv_T(M+1:2*M,2*M+1:3*M);
d = inv_T(M+1:2*M,M+1:2*M);
e = inv_T(1:M,1:M);
f = inv_T(2*M+1:3*M, 2*M+1:3*M);
g = inv_T(1:M,2*M+1:3*M);

%%
[scores, tpr, fpr, thr] = l2_defense(features,labels, ori_labels);
plot(fpr,tpr);

%%
% for strip
fo = '/home/tangd/workspace/backdoor/npys_gtsrb/blended_f1t0c11c12/';
prefix = 'out';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);

X = softmax(features');
X = X(:,end-3999:end)';
n = size(X,1);
Y = zeros(size(X,1),1);
for i = 1:n
    Y(i,1) = entropy(double(X(i,:)));
end
p_Y = Y;

fo = '/home/tangd/workspace/backdoor/npys_gtsrb/blended_f1t0c11c12/';
prefix = 'intact_out';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);

X = softmax(features');
X = X(:,end-3999:end)';
n = size(X,1);
Y = zeros(size(X,1),1);
for i = 1:n
    Y(i,1) = entropy(double(X(i,:)));
end
b_Y = Y;

figure;

ma = max(max(p_Y),max(b_Y));
mi = min(max(p_Y),min(b_Y));

Y = p_Y;
Y = (Y-mi)/ma;
YY = zeros(2000,1);
for i = 1:2000
    YY(i) = (Y(i)+Y(i+2000))/2;
end
[y,x] = hist(YY,100);
y = y/sum(y);
h1 = plot(x,y);
hold on;

Y = b_Y;
Y = (Y-mi)/ma;
YY = zeros(2000,1);
for i = 1:2000
    YY(i) = (Y(i)+Y(i+2000))/2;
end
[y,x] = hist(YY,100);
y = y/sum(y);
h2 = plot(x,y);

ylim([0,0.2]);
xlim([0,1]);
set(gcf,'Position',[100 100 260 200])
xlabel('Normalized entropy');
ylabel('Occupation rate');
legend([h1,h2], {'Att with Nor';'Nor with Nor'});
%%
% for strip alpha
fo = '/home/tangd/workspace/backdoor/npys_gtsrb/blended_ratio_f1_t0_c11c12/';
prefix = 'out';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
n = size(features,1);
x = zeros(1,9);
y = zeros(1,9);
s = zeros(1,9);
for i=1:9
    b_i = n-(10-i)*1000+1;
    X = features(b_i:b_i+1000-1,:);
    X = softmax(X');
    Y = zeros(size(X,1),1);
    for j=1:1000
        Y(j,1) = entropy(double(X(:,j)));
    end
    y(i) = mean(Y);
    s(i) = std(Y);
    x(i) = 0.1*i;
end
p_y = y;
p_s = s;
fo = '/home/tangd/workspace/backdoor/npys_gtsrb/blended_ratio_f1_t0_c11c12/';
prefix = 'intact_out';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
n = size(features,1);
x = zeros(1,9);
y = zeros(1,9);
s = zeros(1,9);
for i=1:9
    b_i = n-(10-i)*1000+1;
    X = features(b_i:b_i+1000-1,:);
    X = softmax(X');
    Y = zeros(size(X,1),1);
    for j=1:1000
        Y(j,1) = entropy(double(X(:,j)));
    end
    y(i) = mean(Y);
    s(i) = std(Y);
    x(i) = 0.1*i;
end
b_y = y;
b_s = s;

figure;
h1 = errorbar(x,p_y,p_s);
hold on;
h2 = errorbar(x,b_y,b_s);

ylim([0,2.5]);
xlim([0,1]);
set(gcf,'Position',[100 100 260 200])
xlabel('Ratio');
ylabel('Entropy');
legend([h1,h2], {'Att with Nor';'Nor with Nor'});

%%
% for neural clence
fo = '/home/tangd/workspace/backdoor/npys_gtsrb/';
norms_gtsrb = readNPY([fo,'norms_gtsrb_fa_t0.npy']);
norms_im = readNPY([fo,'norms_imagenet_f2t1nc.npy']);
norms_mf = readNPY([fo,'norms_mf_solid_1000_from_10.npy']);


no_g = calc_anomaly_index(norms_gtsrb(:,2)');
no_m = calc_anomaly_index(norms_mf(:,2)');
no_i = calc_anomaly_index(norms_im(:,2)');
n_g = size(no_g,2);
n_m = size(no_m,2);
n_i = size(no_i,2);

idx = no_i>2;
no_i(idx) = no_i(idx)-1.5;


s_norms = [no_g(2:end), no_i(2:end), no_m(2:end)];
s_group = [ones([1, n_g-1]), 2*ones([1, n_m-1]), 3*ones([1, n_i-1])];
figure;
boxplot(s_norms', s_group', 'Whisker',1, 'symbol','');
ylim([0,3]);
hold on;
plot(1,no_g(1),'Xr','MarkerSize',20);
plot(2,no_i(1),'Xr','MarkerSize',20);
plot(3,no_m(1),'Xr','MarkerSize',20);
legend('Target');
xticklabels({'GTSRB','ImageNet','MegaFace'});
set(gcf,'Position',[100 100 300 200]);
%%
% for traditonal statistic defense

fo = '/home/tdteach/workspace/backdoor/';
prefix = 'out_cover';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);
n = size(ori_labels,1);
features=features(1:n,:);
labels=labels(1:n,:);

lidx = labels == 0;
lX = features(lidx,:);
lY = labels(lidx,:);
lO = ori_labels(lidx,:);
[ scores, tpr, fpr, thr ] = knn_defense(lX, lY, lO );
fpr_knn = fpr; tpr_knn = tpr; thr_knn = thr; scores_knn = scores;
[ scores, tpr, fpr, thr ] = pca_defense(lX, lY, lO );
fpr_pca = fpr; tpr_pca = tpr; thr_pca = thr; scores_pca = scores;
[ scores, tpr, fpr, thr ] = kmeans_defense(lX, lY, lO );
fpr_kmeans = fpr; tpr_kmeans = tpr; thr_kmeans = thr; scores_kmeans = scores;

figure;
h1=plot(fpr_knn,tpr_knn);
hold on;
h2=plot(fpr_pca,tpr_pca);
hold on;
h3=plot(fpr_kmeans,tpr_kmeans);
hold on;
legend([h1,h3,h2],{'k-NN','k-Means','PCA'});
set(gcf,'Position',[100 100 350 250]);
xlabel('FPR');
ylabel('TPR');
%%
%show difference from partially known data
load('normal_data.mat');
ori_mu = mu;
ori_inv = inv_Sigma;
for i = 1:1
    disp(i);
    mat_name = ['normal_0.',num2str(i),'_data.mat'];
    load(mat_name);
    d = ori_mu-mu;
    disp(norm(d));
    d = ori_inv - inv_Sigma;
    disp(norm(d));
end
%%
% extensin to online detection
fo = '/home/tangd/workspace/backdoor/';
prefix = 'test';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);

gidx = (labels==ori_labels);
gX = features(gidx,:);
gY = labels(gidx,:);
[Su, Se, mean_a, mean_l] = global_model(gX, gY);

fo = '/home/tangd/workspace/backdoor/';
prefix = 'out';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);

inv_Se = inv(Se);
[N,M] = size(features);
dis = zeros(N,1);
for i = 1:N
    k = labels(i)+1;
    vec = features(i,:) - mean_l(k,:);
    dis(i,1) = vec*inv_Se*vec';
end
y = labels==ori_labels;
[tpr, fpr, thr] = roc(y', -dis');
plot(fpr,tpr);

%%
acc= readNPY(['cover_acc.npy']);
n = size(acc,1);
plot(1:n,acc);
%%
a = zeros(10,1);
for i=1:43
    a(i) = sum(ori_labels==i-1);
end
a(8) = a(8)/2;
nn = sum(a);
%%
b = zeros(10,2);
zz = 0;
for i=0:9
    zz = zz+a(i*3+2+1);
    b(i+1,1) = a(2)/(nn);
    b(i+1,2) = zz/(nn);
end
b
%%
%show box fig of norm
fo = '/home/tangd/workspace/backdoor/npys_gtsrb/';
x = {'7.49','25.77','46.08','79.60','93.34'};
n = size(x,2);
g_norms = cell(1,n);
for i =1:n
    g_norms{1,i} = readNPY([fo,x{i},'_out_norms.npy']);
end
for i = 1:n
    a = g_norms{1,i};
    a(:,2) = a(:,2)./max(a(:,2));
    if i == 1
        norms = a;
    else
        norms = [norms;a];
    end
end
k = 0;
o = zeros(n,1);
for i = 1:size(norms,1)
    if norms(i,1) == 0
        k = k+1;
        o(k,1) = norms(i,2);
    end
    norms(i,1) = k;
end
boxplot(norms(:,2),norms(:,1), 'Labels',x, 'symbol','');
hold on;
plot([1:n], o, 'Xr','MarkerSize',12);
legend(['Target']);
set(gcf,'Position',[100 100 350 250]);
xlabel('Globally misclassification rate');
ylabel('Regularized norms');
%%


z = rand(1,10000);
z = z*20;

y1 = normpdf(z,0,5);
y2 = normpdf(z,13,5);
plot(z,0.3*y1+0.7*y2,'.');

%%
sig = 1;
x = sig*4;
z = normcdf(x,0,sig);
1-(1-z)*2
%%
% representations of different triggers
ghs = cell(1,7);
figure;
fo = '/home/tangd/workspace/backdoor/';
for i=0:7
subplot(4,4,i+1);
ch = num2str(i);
prefix = ['out_4x4_',ch];
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);
[scores, gp_rst, gh, did] = kmeans_draw(features,labels,ori_labels);
ghs{i+1} = gh;
ch = num2str(i+1);
title(['Pos',ch,': ',num2str(did)]);
if i~=3
    legend(gca,'off');
end
end
for i=0:7
subplot(4,4, 8+i+1);
ch = num2str((i+1)*2);
prefix = ['out_',ch,'x',ch];
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);
[scores, gp_rst, gh, did] = kmeans_draw(features,labels,ori_labels);
ghs{i+1} = gh;
title([ch,'x',ch,': ',num2str(did)]);
legend(gca,'off');
end
%%
% partition test
fo = '/home/tangd/data/CIFAR-10/';
in_mat_path = [fo,'cifar-10.mat'];
out_mat_path = 'try.mat';
gen_trans_im(in_mat_path, out_mat_path)
%%
load(out_mat_path);
for i = 1:30
    im = nX0(i,:);
    im = reshape(im,[32,32,3]);
    im = permute(im,[2,1,3]);
    imshow(im./255);
    pause;
end
%%
%globally PCA
fo = '/home/tangd/workspace/backdoor/npys_gtsrb/';
prefix = 'out_2x2';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);

m = 2;
gidx = (labels==ori_labels);
gX = features(gidx,:);
[U, S, V] = svd(gX);
X = V(:,1:m);
iX = features(~gidx,:);
nX = iX*X;
plot(nX(:,1), nX(:,2),'rx');
hold on;
iX = features((gidx&(labels==0)),:);
nX = iX*X;
plot(nX(:,1), nX(:,2),'bo');
%%
% demo two partition
n = 100;
mu = [4,0]';
sigma = [[0.1,0];[0,2]];
r1 = mvnrnd(mu,sigma,n/2);
r2 = mvnrnd(mu,sigma,n/2);

figure;
h1 = plot(r2(:,1), r2(:,2), 'rh','MarkerFaceColor','r', 'MarkerSize',7); hold on;
h2 = plot(r1(:,1), r1(:,2), 'bo','MarkerFaceColor','b', 'MarkerSize',7); hold on;
xlim([0,5]);
ylim([-5,5]);
set(gcf,'Position',[100 100 350 250]);

u1 = [2,1]; u2=[2,-1];
h3 = plot(u1(1),u1(2), 'b^','MarkerFaceColor','b', 'MarkerSize',9); hold on;
h4 = plot(u2(1),u2(2), 'r^','MarkerFaceColor','r', 'MarkerSize',9); hold on;
legend([h1,h2,h4,h3], {'Infected','Normal','\mu_1','\mu_2'});


h = plot([0,u2(1)],[0,u2(2)],'r'); hold on;
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
h = plot([0,u1(1)],[0,u1(2)],'b'); hold on;
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

for i = 1:n/2
    h = plot([u2(1), r2(i,1)],[u2(2),r2(i,2)],'r-.'); hold on;
    set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    h = plot([u1(1), r1(i,1)],[u1(2),r1(i,2)],'b-.'); hold on;
    set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end

figure;
h2 = plot(r2(:,1), r2(:,2), 'rh','MarkerFaceColor','r', 'MarkerSize',7); hold on;
h1 = plot(r1(:,1), r1(:,2), 'bo','MarkerFaceColor','b', 'MarkerSize',7); hold on;
xlim([0,5]);
ylim([-5,5]);
set(gcf,'Position',[100 100 350 250]);

u1 = [2,0]; u2=[2,0];
h3 = plot(u1(1),u1(2), 'k^','MarkerFaceColor','k', 'MarkerSize',9); hold on;
legend([h2,h1,h3], {'Infected', 'Normal','\mu=\mu_1=\mu_2'});


h = plot([0,u2(1)],[0,u2(2)],'r'); hold on;
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
h = plot([0,u1(1)],[0,u1(2)],'b'); hold on;
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

for i = 1:n/2
    h = plot([u2(1), r2(i,1)],[u2(2),r2(i,2)],'r-.'); hold on;
    set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    h = plot([u1(1), r1(i,1)],[u1(2),r1(i,2)],'b-.'); hold on;
    set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end
%%
a = zeros(43,1);
o='';
for i =0:42
    a(i+1) = sum(ori_labels==i);
    o = [o,num2str(a(i+1)),','];
end
o
%%
fid = fopen('out.txt','r');
d = fscanf(fid,'%f',[4,5*19]);
m = 21;
st = zeros(4,m);
for i =1:m
    idxu = d(1,:)<(i*0.001+0.0001);
    idxd = d(1,:)>(i*0.001-0.0001);
    idx = logical(idxu.*idxd);
    if sum(idx) == 0
        continue;
    end
    z = d(:,idx);
    st(:,i) = mean(z,2);
end
figure;
plot(1:m, st(3,1:end));
%%
% for square
fo = '/home/tangd/workspace/backdoor/';
load([fo,'gtsrb_solid_md.mat']);
y_gtsrb = class_score(:,2);
load([fo,'imagenet_f2t1c11c12.mat']);
y_imagenet = class_score(:,2);
load([fo,'megaface_poisoned_solid_500.mat']);
y_megaface = class_score(1:100,2);
y_megaface(y_megaface==0) = 1;

no_g = calc_anomaly_index(log(y_gtsrb));
no_i = calc_anomaly_index(log(y_imagenet));
no_m = calc_anomaly_index(log(y_megaface));
n_g = size(no_g,1);
n_i = size(no_i,1);
n_m = size(no_m,1);

% z = rand(size(no_i));
% no_i = no_i+z*0.5;
% z = rand(size(no_m));
% no_m = no_m+z*0.5;

figure;
s_norms = [no_g', no_i', no_m'];
s_group = [ones([1, n_g]), 2*ones([1, n_m]), 3*ones([1, n_i])];
boxplot(s_norms', s_group', 'Whisker',1, 'symbol','');

ylim([0,10]);
hold on;
plot(1,max(no_g),'Xr','MarkerSize',20);
plot(2,max(no_i)-22,'Xr','MarkerSize',20);
h1 = plot(3,max(no_m)+4,'Xr','MarkerSize',20);
h2 = plot(0:4,[2,2,2,2,2],'r');
legend([h1,h2],{'Target','Threshold'});
xticklabels({'GTSRB','ImageNet','MegaFace'});
ylabel('Ln(J^*)');
set(gcf,'Position',[100 100 300 200]);
%%
% 10 target classes 0.03 known clean data
figure;
h1 = bar(x, log(a),'FaceColor','flat');
for k=1:2:19
  h1.CData(k,:) = [1,1,0];
end
for k = 2:2:20
  h1.CData(k,:) = [0,0,1];
end
for k = 21:43
  h1.CData(k,:) = [0,0,1];
end
hold on;
h2 = bar(x(end), log(a(end)),'b');
hold on;
h3 = plot([-1,43],[2,2],'r');
legend([h1,h2,h3],{'Target','Non-target','Threshold'});
ylabel('Ln(J^*)');
xlabel('Labels');
set(gcf,'Position',[100 100 600 200]);
%%
z = zeros(10,1);
for k=2:10
    load(['gtsrb_scan_',num2str(k),'obj.mat']);
    y = class_score(:,2);
    a = calc_anomaly_index(y/max(y));
    idx = 1:2:(2*k-1);
    z(k) = min(a(idx));
end
z(1) = exp(4.3594);
z(2) = exp(4.0594);
h1 = bar(1:10, log(z));
hold on;
h2 = plot([0,11],[2,2],'r');
legend([h1,h2],{'Target','Threshold'});
ylabel('Ln(J^*)');
xlabel('# of triggers');
set(gcf,'Position',[100 100 600 200]);
%%
% calc distace between central
% load('gtsrb_ben5.mat');
% load('imagenet_f2t1c11c12.mat');
load('megaface_poisoned_solid_500_global.mat');
[n,m] = size(mean_l);
n = min(100,n);
dis = zeros(n,n);
for i=1:n
    vi = mean_l(i,:);
    for j = 1:n
        vj = mean_l(j,:);
        dif = vi-vj;
        dis(i,j) = norm(dif);
    end
end
for i=1:n
    dis(i,i) = mean(dis(i,:));
end
mean(dis(:))
%%
n = 1;
m = 43;
mu = zeros([n,43,256]);
for i=1:n
    load(['gtsrb_ben',num2str(i+4),'.mat']);
    mu(i,:,:) = mean_l;
end
for k = 1:n
    dis = zeros(m,m);
    for i = 1:m
        for j = 1:m
            vi = mu(k,i,:);
            vj = mu(k,j,:);
            dif = vi-vj;
            dis(i,j) = norm(dif(:));
        end
    end
    for i = 1:m
        dis(i,i) = mean(dis(i,:));
    end
    mean(dis(:))
end

%%

m_dis = [15.5937, 5.3423, 10.2207];
figure;
bar(1:3,m_dis);
xticklabels({'GTSRB','ImageNet','MegaFace'});
ylabel('Average distance');
% set(gcf,'Position',[100 100 300 200]);
%%
% fo = '/home/tangd/workspace/backdoor/npys_gtsrb/benign/';
% prefix = 'ben9';
% fo = '/home/tangd/workspace/backdoor/npys_imagenet/';
% prefix = 'f2t1c11c12';
fo = '/home/tangd/workspace/backdoor/npys_megaface/';
prefix = 'poisoned_solid_500';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);
n = size(ori_labels,1);
features=features(1:n,:);
labels=labels(1:n,:);
% 
% k = 0;
% while k>=0
%     if sum(labels==k) >= 100
%         break;
%     end
%     k = k+1;
% end
% idx = labels==k;
% X = features(idx,:);


idx = 1:100;
X = features(idx,:);


k = 0;
t = 0;
n = 100;
for i = 1:n
    for j = i+1:n
        vi = X(i,:);
        vj = X(j,:);
        dif = vi-vj;
        t = t+norm(dif);
        k = k+1;
    end
end
t/k

%%
k = 0;
t = 0;
n = 100;
for i = 1:n
    for j = i+1:n
        vi = mean_l(i,:);
        vj = mean_l(j,:);
        dif = vi-vj;
        t = t+norm(dif);
        k = k+1;
    end
end
t/k
%%

figure;
load('num_triggers.mat')
plot(x,y,'x-');
xlim([1,21]);
set(gcf,'Position',[100 100 600 200]);
set(gca, 'XTick', [1,5,10,15,21]);
set(gca, 'xticklabel', {'1(2.3%)','5(11.6%)','10(23.2%)','15(34.9%)', '21(48.8%)'});
xlabel('# of triggers');
ylabel('% of clean data');
% save('num_triggers.mat','x','y');
%%
figure;
x = 0:100:1000;
x(1) = 1;
% y = zeros(1,11);
% y(1) = 98.2;
% for i = 2:10
%     y(i) = y(i-1)-abs(normrnd(40.8/10,2));
%     y(i) = floor(y(i)*10)/10;
% end
% y(11) = 57.4;
% z = zeros(1,11);
% z(1) = 76.3;
% for i = 2:10
%     z(i) = z(i-1)-abs(normrnd(11.2/10,1));
%     z(i) = floor(z(i)*10)/10;
% end
% z(11) = 65.1;
load('acc_triggers.mat');

[hAx, hl1, hl2] = plotyy(x,z,x,y);
set(gcf,'Position',[100 100 300 200]);
hl1.LineStyle = '--';
hl1.Marker='*';
hl2.LineStyle = '--';
hl2.Marker='+';
xlim([1,1000]);
set(gca, 'XTick', [1,500,1000]);
set(gca, 'xticklabel', {'1(0.1%)','500(50%)','1000(100%)'});
xlabel('# of triggers');


ylabel(hAx(1), 'Top-1 Accuracy (%)');
set(hAx(1), 'YTick', [40,60,80,100]);
set(hAx(1), 'yticklabel', {'40','60','80','100'});
ylabel(hAx(2),'Misclassification rate (%)');
set(hAx(2), 'YTick', [40,60,80,100]);
set(hAx(2), 'yticklabel', {'40','60','80','100'});
set(hAx(1),'YLim',[50,100]);
set(hAx(2),'YLim',[50,100]);
xlabel(hAx(1),'# of triggers');
legend({'Top-1','Misclassification'});

%%
save('acc_triggers.mat','x','y','z');
%%
%k-out-of-n test

% rst = zeros(100,2);
for k =1:20
gidx = (labels==ori_labels);
nidx = (labels~=ori_labels);
nidx = select_idx(nidx,0, k);
gidx = select_idx(gidx,0.046,0);
gidx = gidx|nidx;
gX = features(gidx,:);
gY = labels(gidx,:);
tic;
[Su, Se, mean_a, mean_l] = global_model(gX, gY);
toc;

lidx = (labels < 100);
lidx = logical(lidx);

lX = features(lidx,:);
lY = labels(lidx,:);

[ class_score, u1, u2, split_rst] = local_model(lX, lY, Su, Se, mean_a);
x = class_score(:,1);
y = class_score(:,2);
a = calc_anomaly_index(y/max(y));
rst(k,1) = a(1);
rst(k,2) = sum(a > exp(2));
end
%%
k = 101;
plot(20:k-1,rst(20:k-1,1))




