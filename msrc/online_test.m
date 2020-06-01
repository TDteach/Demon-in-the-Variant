home_folder = getenv('HOME');
fo = fullfile(home_folder,'/data/npys/backdoor');
mat_folder = fullfile(home_folder,'/data/mats/backdoor');

%%
fn = 'gtsrb_s1_t0_c23_f1';
[features,labels,ori_labels] = read_features(fn,fo);
rX = features; rY = labels; rP = ~(labels==ori_labels);

%%
gidx = (labels==ori_labels);
bX = features(gidx,:); bY = labels(gidx,:);
[mX, mY, rX, rY] = select_k_from_every_class(bX,bY,100);
gb_model = global_model(mX, mY);
lc_model = local_model(mX,mY,gb_model,true);
x = lc_model.sts(:,1);
y = lc_model.sts(:,2);
ai = calc_anomaly_index(y/max(y)); 
ai
%%
ratio = 0.1;
gidx = (labels==ori_labels);
bX = features(gidx,:); bY = labels(gidx,:);
tidx = rand(size(bY));
midx = tidx>(1-ratio); ridx = (~midx);
mX = bX(midx,:); mY = bY(midx,:);
rX = bX(ridx,:); rY = bY(ridx,:);
gb_model = global_model(mX, mY);
lc_model = local_model(mX,mY,gb_model,true);
x = lc_model.sts(:,1);
y = lc_model.sts(:,2);
ai = calc_anomaly_index(y/max(y)); 
ai

%%
bidx = (~gidx);

tX = features(bidx,:); tY = labels(bidx,:);
rP = zeros(size(rY)); tP = ones(size(tY));
rX = [rX;tX]; rY = [rY;tY]; rP = [rP;tP];
%%
[N,M] = size(rX); 
sfx = randperm(N);
rX = rX(sfx,:); rY = rY(sfx,:); rP = rP(sfx,:);

up_model = lc_model;
rst_sc = zeros(N,1);
for i=1:N
    disp(i);
    x = rX(i,:);
    y = rY(i,1);
    [up_model] = online_update(x,y,gb_model, up_model);
    
    xx = up_model.sts(:,1);
    yy = up_model.sts(:,2);
    ai = calc_anomaly_index(yy/max(yy)); 
    
    rst_sc(i) = ai(up_model.lb_map(y));
end

%%
save('cifar10_s0_t7_c12_400-ratio.mat','rst_sc','rX','rY','rP','gb_model','lc_model','up_model');
%%
pidx = rP>0.5;
nidx = (~pidx);
pp = sum(pidx); nn=sum(nidx);
py = rst_sc(pidx); ny = rst_sc(nidx);
thr = exp(2);
tp = sum(py>=thr); tn = sum(ny<thr);
fn = sum(py<thr);  fp = sum(ny>=thr);
tpr = tp/pp
fpr = fp/nn
%%
[tpr,fpr,thr] = roc(rP',rst_sc');
%%
tgt_tpr = [0.95,0.995,0.999];
tgt_fpr = [0.0005,0.001,0.005];
m = size(tgt_tpr,2);
nd_tpr = -ones(1,m);
nd_fpr = -ones(1,m);
nd_thr = -ones(1,m);
for i=1:size(tpr,2)
    for j=1:m
        if (tpr(i) >= tgt_tpr(j)) && ((nd_fpr(j) < 0)||(fpr(i) < nd_fpr(j)))
            nd_tpr(j) = tpr(i);
            nd_fpr(j) = fpr(i);
            nd_thr(j) = thr(i);
        end
    end
end
disp(nd_tpr);
disp(nd_fpr);
disp(nd_thr);
%%
plot(fpr(1:3000),tpr(1:3000));
%%
%STRIP
replica = 10;
home_folder = getenv('HOME');
fo = fullfile(home_folder,'/data/npys/backdoor');
mat_folder = fullfile(home_folder,'/data/mats/backdoor');

% fn = 'gtsrb_1_strip_benign';
fn = 'out';
[features,labels,ori_labels] = read_features(fn,fo);
%%
logX = log2(features);
H = logX.*features;
H = -sum(H,2);
N = size(features,1)/replica;

h_sc = zeros(N,1);
p_lb = zeros(N,1);
for i=1:N
    k = i*replica;
    h_sc(i) = mean(H(k-replica+1:k));
    p_lb(i) = ~(labels(k)==ori_labels(k));
end
[tpr,fpr,thr] = roc(p_lb',h_sc');
plot(fpr,tpr);
%%
N = size(features,1)/replica;
M = size(features,2);

a_ft = zeros(N,M);
h_sc = zeros(N,1);
p_lb = zeros(N,1);
for i=1:N
    k = i*replica;
    a_ft(i,:) = mean(features(k-replica+1:k, :));
    p_lb(i) = ~(labels(k)==ori_labels(k));
    h_sc(i) = -a_ft(i,:) * log2(a_ft(i,:))';
end
[tpr,fpr,thr] = roc(p_lb',h_sc');
plot(fpr,tpr);
%%
[v,a] = max(features,[],2);
sum(a==(labels+1))