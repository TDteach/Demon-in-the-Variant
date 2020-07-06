home_folder = getenv('HOME');
fo = fullfile(home_folder,'workspace/backdoor/neural_cleance/logs/cifar10_walk_no_cover/');
files = dir(fo);

num_classes = 43;
if contains(fo,'cifar10')
    num_classes = 10;
elseif contains(fo,'gtsrb')
    num_classes = 43;
elseif contains(fo,'imagenet')
    num_classes = 1001;
elseif contains(fo,'megaface')
    num_classes = 647608;
end

scs = zeros(2,1);
tgt = zeros(2,1);
nsc = 0;
for i = 1:size(files,1)
    na = files(i).name;
    if contains(na,'checkpoint') == 0
        continue;
    end
    
    pos = strfind(na,'_');
    tid = str2num(na(pos(2)+2:pos(3)-1));
    
    fpath = fullfile(fo,na);
    fid = fopen(fpath); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    val = jsondecode(str);
    
    a = zeros(1,num_classes);
    for k=0:num_classes-1
        a(k+1) = getfield(val,['x',num2str(k)]);
    end
    ind = calc_anomaly_index(a);
    for k=0:num_classes-1
        nsc = nsc+1;
        scs(nsc) = ind(k+1);
        if (k == tid)
            tgt(nsc) = 1;
        else
            tgt(nsc) = 0;
        end
    end
end


[tpr,fpr,thr] = roc(tgt',scs');

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
sum(scs(idx)>2)
sum(scs(idx)<=2)
idx = (tgt==0);
sum(scs(idx)>2)
sum(scs(idx)<=2)

