addpath(genpath('msrc'))
%%
folder='npys';
N = 16;
M = 256;
features = cell(1,N);
for k=0:N
    d = readNPY([folder,'/data_',num2str(k),'_0.npy']);
    features{k+1} = d;
end
labels = readNPY([folder,'/label.npy']);
%%
N = 16;
M = 256;
% fo = '/home/tangd/workspace/backdoor/npys/gtsrb_t0f1c11c12/';
fo = '/home/tangd/workspace/backdoor/';
features = readNPY([fo,'out_X.npy']);
labels = readNPY([fo,'out_labels.npy']);
% ori_labels = readNPY(['/home/tangd/workspace/backdoor/','ori_labels.npy']);
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
uni_lbs = unique(labels);
L = size(uni_lbs,1);
cnt_l = zeros(max(labels)+1,1);
n = size(labels,1);
for i=1:n
    k = labels(i);
    cnt_l(k+1) = cnt_l(k+1)+1;
end
%%
zz=zeros(100000,1);
for i=1:100000
    zz(i) = sum(cnt_l(i:i+5-1));
end
[a,b] = sort(zz);

%%
N=16;
load('benign_#51_good');
benign_features = features;
benign_labels = labels;
load('benign_#51_normal_lu');
n = size(coss,1);
for i = 1:n
    if (benign_labels(i) < 3000) || (benign_labels(i) > 3100)
        continue;
    end
    for j = 1:N+1
        benign_features{j}(i,:) = features{j}(i,:);
    end
    benign_labels(i) = labels(i);
end
features = benign_features;
labels = benign_labels;
%%
N = 16;
load('msrc/poisoned_normal_lu_#51_good');
poisoned_features = features;
poisoned_labels = labels;
load('msrc/poisoned_normal_lu_#51_bad');
n = size(coss,1);
for i = 1:n
    if (poisoned_labels(i) < 8994-1) || (poisoned_labels(i) > 8994-1)
%     if (poisoned_labels(i) < 3000) || (poisoned_labels(i) > 3100)
        continue;
    end
    for j = 1:N+1
        poisoned_features{j}(i,:) = features{j}(i,:);
    end
    poisoned_labels(i) = labels(i);
end
features = poisoned_features;
labels = poisoned_labels;
%%
poisoned_coss = coss;
%%
benign_coss = coss;

%%
M = 256;
N = 16;
n = size(labels,1);
uni_lbs = unique(labels);
L = size(uni_lbs,1);
rank_l = zeros(n,1);
[y,I] = sort(labels);

k = 1;
for i=1:n
    if (i>1) && (y(i) ~= y(i-1))
        k = k+1;
    end
    rank_l(I(i)) = k;
end

cnt_l = zeros(L,1);
mid_features = zeros(L, M);

%calc mean
for i =1:n
    vec = features{N+1}(i,:);
    %vec = vec/norm(vec);
    k = rank_l(i);
    mid_features(k,:) = mid_features(k,:)+vec;
    cnt_l(k) = cnt_l(k)+1;
end
for k =1:L
    mid_features(k,:) = mid_features(k,:)/cnt_l(k); 
    mid_features(k,:) = mid_features(k,:)/norm(mid_features(k,:));
end

%calc coss
coss = zeros(n,N+1);
for i = 1:n  
    k = rank_l(i);
    mid = mid_features(k,:);
    for j=1:N
        vec = features{j}(i,:);
        coss(i,j) = vec*mid'/norm(vec);
    end
    j = N+1;
    vec = features{j}(i,:);
    coss(i,j) = vec*mid'/norm(vec);
end

ms=mean(coss);
ss=zeros(1,N);
for j = 1:N
    ss(j) = std(coss(:,j));
end

figure;
x = 1:N+1;
plot(x,ms);
ylim([0,1])
%%
save('poisoned_normal_lu_#51_good','features','labels','coss','mid_features');

%%
% count labels 
max_l = max(labels);
lb_ct = zeros(max_l+1,1);
n = size(labels,1);
for i = 1:n;
    lb_ct(labels(i)+1) = lb_ct(labels(i)+1)+1;
end

% for 21-list
% max is 666 with label 5449
% second is 397 with label 6279
% for 51-list
% max is 468 with label 3537
% second is 423 with label 5455

%%
cc=0;
bad_list = zeros(2,1);
test = zeros(1,N);
show_rst = 1;

if show_rst > 0
    f1 = figure;
    if show_rst > 1
        f2 = figure;
    end
end


for i = 1:n
% for i = 2803:n
%     i = rank_l(j); 
%     if labels(i) == 4772
     if labels(i) == 0;
        
        
        k = rank_l(i);
        mid = mid_features(k,:);
        
        for j = 1:N
            vec = features{j}(i,:);
            test(1,j) = vec*mid'/norm(vec);
        end
     
        if mean(test) < 1
            i
            cc = cc+1;
            bad_list(cc,1) = i;

            if show_rst > 0
                figure(f1);
                plot(x, test)
                ylim([0,1])
                if show_rst > 1
                    figure(f2);
                    im = imread(img_path{i});
                    imshow(im);
                end
                pause;
            end
        end
    end
    
end
cc
%%
%check difference
%bad_list - poisoned_list => diff_list

diff_list = zeros(2,1);
dk = 0;
j = 1;

n_p = 0;
n_n = 0;
for i=1:size(bad_list,1)
    
    while bad_list(i) > poisoned_list(j)
        dk = dk+1;
        diff_list(dk) = -poisoned_list(j);
        n_n = n_n+1;
        j = j+1;
        if j > size(poisoned_list,1)
            break;
        end
    end
    
    if j > size(poisoned_list,1)
        break;
    end
    
    if bad_list(i) == poisoned_list(j)
        j = j+1;
    else 
        dk = dk+1;
        diff_list(dk) = bad_list(i);
        n_p = n_p+1;
    end
end

for ii = i:size(bad_list,1)
    dk = dk+1;
    diff_list(dk) = bad_list(ii);
    n_p = n_p+1;
end
%%
% see the differences between benign and poisoned model
load('benign_#51_bad.mat');
benign_coss = coss;
load('poisoned_normal_#51_bad.mat');
poisoned_coss = coss;
%%
% see differences between curves
N = 16;
M = 256;
n = size(coss,1);
x = 1:N+1;
for i = 1:n
    
    if idx_int(i) == 0 
        continue;
    end
    
    
    display(['idx: ',num2str(i)])
    display(['labels: ',num2str(benign_labels(i))])
    display(['score: ',num2str(score(i))])
    
    benign_labels(i)
    f1 = figure;
    im = imread(img_path{i});
    imshow(im);
    
    
    f2 = figure;
    plot(x,benign_coss(i,:));
    hold on;
    plot(x,poisoned_coss(i,:));
    hold on;
    
    ylim([0,1]);
    legend({'benign','poisoned','s_b','s_p'});
    pause;
    close(f1);
    close(f2);
end
%%
zuo = zeros(n,N);
for j = 1:N
    zuo(:,j) = benign_coss(:,j)-benign_coss(:,j+1);
end
MU = mean(zuo);
SIGMA = cov(zuo);
INV_SIGMA = inv(SIGMA);

%%
var_b = benign_coss(:,1:N)-repmat(benign_coss(:,N+1),[1,N]);
MU = mean(var_b);
SIGMA = cov(var_b);
INV_SIGMA = inv(SIGMA);

%%
diff_coss = benign_coss-poisoned_coss;
MU = mean(diff_coss);
SIGMA = cov(diff_coss);
INV_SIGMA = inv(SIGMA);
%%
p = zeros(n,1);
for i = 1:n
    dif = diff_coss(i,:);
    p(i,1) = dif*INV_SIGMA*transpose(dif);
end
%%
pb = zeros(n,1);
pp = zeros(n,1);
for i = 1:n
    dif = poisoned_coss(i,1:N)-poisoned_coss(i,N+1) - MU;
    pp(i,1) = dif*INV_SIGMA*transpose(dif);
    dif = benign_coss(i,1:N)-benign_coss(i,N+1) - MU;
    pb(i,1) = dif*INV_SIGMA*transpose(dif);
end
diff_p = abs(pb-pp);
%%
p_zuo = zeros(n,1);
for i = 1:n
    dif= zeros(1,N);
    for j = 1:N
        dif = poisoned_coss(i,j)-poisoned_coss(i,j+1);
    end
    dif = dif-MU;
    pp(i,1) = dif*INV_SIGMA*transpose(dif);
    
    dif= zeros(1,N);
    for j = 1:N
        dif = benign_coss(i,j)-benign_coss(i,j+1);
    end
    dif = dif-MU;
    pb(i,1) = dif*INV_SIGMA*transpose(dif);
end
diff_p = abs(pb-pp);

%%

ylim([-1,1]);
hold on;
plot(1:N,MU);
hold on;
%%
%calc Su Se

subm = coss(1:n,1:N);
for i = 1:n
    for j=1:N
        subm(i,j) = poisoned_coss(i,j) - poisoned_coss(i,j+1);
    end
end

covm = zeros(N,N);
for i = 1:n
    z = subm(i,1:N);
    cc = z'*z;
    covm = covm+cc;
end
covm = covm./(n-1);
mv = mean(subm);
covm = covm-mv'*mv;
inv_covm = inv(covm);
%%
%calc mahalanobis distance 
ma_dist = zeros(n,1);
for i =1:n
    z = subm(i,1:N);
    z = z-mv;
    ma_dist(i,1) = z*inv_covm*z';
end

%%
[Su, Se, A, G, score,u,e] = joint_bayesian(features, labels);
% [Su, Se, A, G, score,u,e] = joint_bayesian(features, labels, good_Su, good_Se);
%%
good_Se = Se;
good_Su = Su;
good_u = u;
good_e = e;
%%
% save('good_rst_poisoned_normal_lu_#51_8993','good_Su','good_Se','good_u','good_e');
%%
hist(score,10000)    
idx_zero = (labels==0);
idx_int = (score<0);
display(['# hits: ', num2str(sum(idx_zero.*idx_int))])
display(['# prob: ', num2str(sum(idx_int))])
display(['# true: ', num2str(sum(idx_zero))])
%%
[dist_Se, norm_e, dist_u, mean_f, dist_r] = test_Se(good_u,good_e,labels,Su,Se);
hist(dist_Se,10000) 
%%
[ class_score, u1, u2, split_rst] = EM_like(features, labels, Su, Se, A, G);
%%
show_score = class_score(1:10)
plot(show_score/max(show_score))
%%
a = calc_anomaly_index(show_score)
plot(a);
%%

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
% for ac
[scores] = kmeans_defense(features,labels);
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
norms_gtsrb = readNPY(['/home/tangd/workspace/backdoor/','norms_gtsrb_fa_t0.npy']);
norms_mf = readNPY(['/home/tangd/workspace/backdoor/','norms_mf_solid_1000_from_100.npy']);


no_g = calc_anomaly_index(norms_gtsrb(:,2)');
no_m = calc_anomaly_index(norms_mf(:,2)');
no_i = no_m;
n_g = size(no_g,2);
n_m = size(no_m,2);
n_i = size(no_i,2);


s_norms = [no_g(2:end), no_m(2:end), no_i(2:end)];
s_group = [ones([1, n_g-1]), 2*ones([1, n_m-1]), 3*ones([1, n_i-1])];

boxplot(s_norms', s_group', 'Whisker',1, 'symbol','');
ylim([0,5]);
hold on;
plot(1,no_g(1),'Xr','MarkerSize',20);
plot(2,no_m(1),'Xr','MarkerSize',20);
plot(3,no_i(1),'Xr','MarkerSize',20);

xticklabels({'GTSRB','MegaFace','ImageNet'})