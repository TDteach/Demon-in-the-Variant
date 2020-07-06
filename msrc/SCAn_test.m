home_folder = getenv('HOME');
fo = fullfile(home_folder,'data/npys/backdoor/cifar10_walk_no_cover/');
files = dir(fo);

num_classes = 10;

scs = zeros(2,1);
tgt = zeros(2,1);
nsc = 0;
z = 0;
itk = 1;
for i = 1:size(files,1)
    na = files(i).name;
    if contains(na,'checkpoint') == 0
        continue;
    end
    if contains(na,'normal')
        continue;
    end
    if contains(na,'X') == 0
        continue;
    end
    pos = strfind(na,'_');
    
    
    disp(na);
    tid = str2num(na(pos(2)+2:pos(3)-1));
    
    fn = na(1:pos(4)-1);
    [features,labels,ori_labels] = read_features(fn,fo);
    [gb_model, lc_model, ai] = SCAn(features, labels, ori_labels, 0.1);
    
    for k=0:num_classes-1
        nsc = nsc+1;
        scs(nsc) = ai(k+1);
        if (k == tid)
            tgt(nsc) = 1;
        else
            tgt(nsc) = 0;
        end
    end
end

save('SCAn_cifar10_no_cover','scs','tgt');


[tpr,fpr,thr] = roc(tgt',scs');

tgt_tpr = [0.95,0.990,0.999];
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