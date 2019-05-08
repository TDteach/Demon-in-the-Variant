addpath(genpath('msrc'))
%%
N = 16;
M = 256;
% fo = '/home/tangd/workspace/backdoor/npys_gtsrb/benign/';
fo = '/home/tangd/workspace/backdoor/';
features = readNPY([fo,'out_X.npy']);
labels = readNPY([fo,'out_labels.npy']);
ori_labels = readNPY([fo,'ori_labels.npy']);
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
fo = '/home/tangd/workspace/backdoor/';
features = readNPY([fo,'out_X.npy']);
labels = readNPY([fo,'out_labels.npy']);
ori_labels = readNPY([fo,'ori_labels.npy']);

[crt_Su, crt_Se, crt_mean_a] = global_model(features, labels);
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
Se = crt_Se;
mu = crt_mu;
inv_Sigma = inv(Se);
save(['normal_1.0_data.mat'],'inv_Sigma','mu');
%%
r = 9; k = 1;
Se = rst_Se{r,k};
mu = rst_mu{r,k};
inv_Sigma = inv(Se);
save(['normal_0.',num2str(r),'_data.mat'],'inv_Sigma','mu');
% save('normal_1.0_data.mat','inv_Sigma','mu');
% save('good_rst_poisoned_normal_lu_#51_8993','good_Su','good_Se','good_u','good_e');
%%
fo = '/home/tangd/workspace/backdoor/';
prefix = 'out_2x2';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);

% ori_labels(ori_labels<3) = 0;
% labels(labels<3) = 0;
%%
% global model
gidx = (labels==ori_labels);
% gidx = (labels>=0);
c = rand(size(gidx));
gidx = gidx.*c;
gidx = gidx>(1-0.8);
gX = features(gidx,:);
gY = labels(gidx,:);
% gidx = ~rst_idx{r,k};
% gidx = (gY >= 10);
% gX = gX(gidx,:);
% gY = gY(gidx,:);
[Su, Se, mean_a, mean_l] = global_model(gX, gY);
%%
%local model
lidx = (labels<10);
% lidx = lidx.*(labels==ori_labels);
lidx = logical(lidx);

lX = features(lidx,:);
lY = labels(lidx,:);
[ class_score, u1, u2, split_rst] = local_model(lX, lY, Su, Se, mean_a);

x = class_score(:,1);
y = class_score(:,2);
figure;
plot(x, y/max(y));
hold on;
a = calc_anomaly_index(y/max(y));
plot(x, a);
figure;
n = size(u1,1);
dis_u = zeros(n,1);
F = inv(Se);
for i=1:n
    d = u1(i,:)-u2(i,:);
    dis_u(i,1) = d * F * d';
end
b = log(det(Se));
dis_u = dis_u+b;
plot(x,dis_u);

%%

rst = zeros(9,9);
for r = 1:9
    for k = 1:9
      [Su, Se, mean_a, class_score] = our_defense(features, labels, ori_labels, r*0.1);
      rst(r,k) = max(class_score(:));
    end
end

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
% for ss
fo = '/home/tangd/workspace/backdoor/';
prefix = 'out_4x4_0';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);
[scores, s0, v0] = ss_defense(features, labels,ori_labels);
% 
% figure;
% hist(scores);

%%
% for ac
fo = '/home/tangd/workspace/backdoor/';
prefix = 'out_2x2';
features = readNPY([fo,prefix,'_X.npy']);
labels = readNPY([fo,prefix,'_labels.npy']);
ori_labels = readNPY([fo,prefix,'_ori_labels.npy']);
[scores] = kmeans_defense(features, labels,ori_labels);
figure;
boxplot(scores(:,1), scores(:,2), 'PlotStyle','compact','symbol','.');
%%
% for ac
ylim([-0.5,1]);
set(gcf,'Position',[100 100 1000 200])
xlabel('label');
ylabel('silhouette score');
% hist(scores)

%%
% for SentiNet
idx_3 = labels==3;
X = features(idx_3,:);
[a,b] = max(X');
xv = sum(b==1)/size(b,2);
idx_7 = labels==7;
X = features(idx_7,:);
y = softmax(X');
size(y)
yv = y(7+1,:);
plot(yv,xv*ones(1,size(yv,2)), '.');


%%
% for strip
figure;
X = softmax(features');
X = X(:,end-3999:end)';
n = size(X,1);
Y = zeros(size(X,1),1);
for i = 1:n
    Y(i,1) = entropy(double(X(i,:)));
end
%%
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
plot(x,y);
hold on;

Y = b_Y;
Y = (Y-mi)/ma;
YY = zeros(2000,1);
for i = 1:2000
    YY(i) = (Y(i)+Y(i+2000))/2;
end
[y,x] = hist(YY,100);
y = y/sum(y);
plot(x,y);

ylim([0,0.2]);
xlim([0,1]);
set(gcf,'Position',[100 100 260 200])
xlabel('Normalized entropy');
ylabel('Occupation rate');
legend({'Infected';'Intact'});
%%
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
errorbar(x,y,s);
%%
figure;
errorbar(x,p_y,p_s);
hold on;
errorbar(x,b_y,b_s);

ylim([0,2.5]);
xlim([0,1]);
set(gcf,'Position',[100 100 260 200])
xlabel('Ratio');
ylabel('Entropy');
legend({'Infected';'Intact'});
%%
% for sentinet
figure;
n = 300;
m = 100;
a = softmax(features');
[v_max, a_max] = max(a); 
fool = zeros(n,1);
avg = zeros(n,1);
for i=1:n
    for j=(i-1)*m*2+1:2:i*m*2
        if a_max(j) == labels(j)+1
            fool(i) = fool(i)+1;
        end
%         fool(i) = fool(i)+a(labels(j)+1,j);
        avg(i) = avg(i)+v_max(j+1);
    end
end
fool = fool/m;
avg = avg/m;
plot(avg(1:100),fool(1:100),'^r');
hold on;
plot(avg(101:300),fool(101:300),'.b');

ylim([0,1]);
xlim([0,1]);
set(gcf,'Position',[100 100 300 200])
xlabel('AvgConf');
ylabel('Fooled');
legend({'Infected';'Intact'});

%%
% for neural clence
fo = '/home/tangd/workspace/backdoor/npys_gtsrb/';
norms_gtsrb = readNPY([fo,'norms_gtsrb_f1_t0_c11c12.npy']);
norms_mf = readNPY([fo,'norms_mf_solid_1000_from_10.npy']);
norms_im = readNPY([fo,'norms_imagenet_f1_t0_nc_uniform.npy']);


no_g = calc_anomaly_index(norms_gtsrb(:,2)');
no_m = calc_anomaly_index(norms_mf(:,2)');
no_i = calc_anomaly_index(norms_im(:,2)');
n_g = size(no_g,2);
n_m = size(no_m,2);
n_i = size(no_i,2);


s_norms = [no_g(2:end), no_m(2:end), no_i(2:end)];
s_group = [ones([1, n_g-1]), 2*ones([1, n_m-1]), 3*ones([1, n_i-1])];

boxplot(s_norms', s_group', 'Whisker',1, 'symbol','');
ylim([0,3]);
hold on;
plot(1,no_g(1),'Xr','MarkerSize',20);
plot(2,no_i(1),'Xr','MarkerSize',20);
plot(3,no_m(1),'Xr','MarkerSize',20);
legend('target class');
xticklabels({'GTSRB','ImageNet','MegaFace'});
set(gcf,'Position',[100 100 300 200]);
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
legend(['target class']);
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
sig = 2;
x = sig*3;
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
[scores, gp_rst, gh, did] = kmeans_defense(features,labels,ori_labels);
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
[scores, gp_rst, gh, did] = kmeans_defense(features,labels,ori_labels);
ghs{i+1} = gh;
title([ch,'x',ch,': ',num2str(did)]);
legend(gca,'off');
end
%%

fo = '/home/tangd/data/CIFAR-10/';
load([fo,'cifar-10.mat']);
gX = double(labels);
gY= images';

[coeff, XX] = pca(gX);

%%
[Su, Se, mean_a, mean_l] = global_model(XX(:,1:100), gY);


