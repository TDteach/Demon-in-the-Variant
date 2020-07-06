home_folder = getenv('HOME');
fo = fullfile(home_folder,'/data/npys/backdoor');
% fn = 'gtsrb_100_strip_s1_t0_c23_f1_trigger2';
fn = 'cifar10_100_strip_s0_t1_c23_l0_1r_solid';
% fo = fullfile(home_folder,'/data/npys/');
% fn = 'out';
[features,labels,ori_labels] = read_features(fn,fo);

%EntropySum = -np.nansum(py1_add*np.log2(py1_add))

strip_N = 100;
[N,M] = size(features);

scs = zeros(2,1); tgt = zeros(2,1);
k = 0;
for i =1:strip_N:N
    k = k+1;
    X = features(i:i+strip_N-1,:);
    z = X.*log2(X);
    scs(k) = -mean(nansum(z,2));
    if labels(i) == ori_labels(i)
        tgt(k) = 0;
    else
        tgt(k) = 1;
    end
end

[tpr,fpr,thr] = roc(tgt',-scs');

tgt_tpr = [0.95,0.995,0.999];
tgt_fpr = [0.0005,0.001,0.005];
m = size(tgt_tpr,2);
nd_tpr = -ones(1,m);
nd_fpr = -ones(1,m);
nd_thr = -ones(1,m);
for i=1:size(tpr,2)
    for j=1:m
%         if (fpr(i) <= tgt_fpr(j)) && ((nd_fpr(j) < 0)||(tpr(i) > nd_tpr(j)))
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

idx = (tgt==1);
[y,x] = hist(scs(idx),20);
% h1 = plot(x,100*y/sum(idx),'r');
h1 = bar(x,100*y/sum(idx),'r');
idx = (tgt==0);
[y,x] = hist(scs(idx),20);
hold on;
% h2 = plot(x,100*y/sum(idx),'b');
h2 = bar(x,100*y/sum(idx),'b');
legend([h1,h2],{'Att over Nor','Nor over Nor'});
xlabel('Entropy');
ylabel('Percentage');
set(gcf,'Position',[100 100 260 200])
