home_folder = getenv('HOME');
fo = fullfile(home_folder,'data/npys/backdoor/');

prefix = 'gtsrb_sa_t0_cn_f0.1_trigger2';

pos = strfind(prefix,'_');
target_lb = str2num(prefix(pos(2)+2:pos(3)-1));

fn = [prefix,'_clean'];
[features,labels,ori_labels] = read_features(fn,fo);
gb_model = global_model(features, labels);
lc_model = local_model(features,labels,gb_model,true);
x = lc_model.sts(:,1);
y = lc_model.sts(:,2);
ai = calc_anomaly_index(y/max(y)); 
ai
%%
% fo = fullfile(home_folder,'data/npys/');
% fn = 'out';
fn = [prefix,'_test'];
[rX,rY,oY] = read_features(fn,fo);
o_idx = true(size(rY));
for i=1:size(oY,1);
    if (oY(i) == target_lb)
        o_idx(i) = false;
    end
end

rX = rX(o_idx); rY = rY(o_idx); oY = oY(o_idx);
rP = (rY~=oY);
%%
lb_cc = zeros(43,1);
for k = 0:42
    lb_cc(k+1) = sum(oY==k);
end
%%

[N,M] = size(rX); 
sfx = randperm(N);
rX = rX(sfx,:); rY = rY(sfx,:); rP = rP(sfx,:); oY = oY(sfx,:);

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
save('gtsrb_sa_t0_c23_online.mat','rst_sc','rX','rY','rP','gb_model','lc_model','up_model');
%%
[tpr,fpr,thr] = roc(rP',rst_sc');

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