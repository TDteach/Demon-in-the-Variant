home_folder = getenv('HOME');
fo = fullfile(home_folder,'/data/npys/');
mat_folder = fullfile(home_folder,'/data/mats/backdoor');
replica = 500;

fn = 'out';
[features,labels,ori_labels] = read_features(fn,fo);
[N,M] = size(features);
if mod(N,replica*2) > 0
    N = N-mod(N,replica*2);
    features=features(1:N,:); labels=labels(1:N,:); ori_labels=ori_labels(1:N,:);
end


[cof_Y, prd_Y] = max(features,[],2);
prd_Y = prd_Y-1;

[N,M] = size(features);
ip_idx = 2:2:N;
xx_idx = 1:2:N;
ip_Y = prd_Y(ip_idx,:); ip_F = cof_Y(ip_idx,:); ip_lb = labels(ip_idx,:); ip_ori = ori_labels(ip_idx,:);
xx_Y = prd_Y(xx_idx,:); xx_F = cof_Y(xx_idx,:); xx_lb = labels(xx_idx,:); xx_ori = ori_labels(xx_idx,:);

n = int32(N/replica/2);
fooled = zeros(n,1);
avgconf = zeros(n,1);
poed = zeros(n,1);
for i=1:n
    r = i*replica; l = r-replica+1;
    poed(i) = (xx_lb(l) ~= xx_ori(l))*1.0;
    fooled(i) = sum(xx_Y(l:r)==xx_lb(l))/replica;
    avgconf(i) = mean(ip_F(l:r));
end

po_idx = (poed>0.5); bn_idx = (~po_idx);
po_x = avgconf(po_idx); po_y = fooled(po_idx);
bn_x = avgconf(bn_idx); bn_y = fooled(bn_idx);
plot(po_x,po_y,'^r');
hold on;
plot(bn_x,bn_y,'.b');

ylim([0,1]);
xlim([0,1]);
set(gcf,'Position',[100 100 400 200])
xlabel('AvgConf');
ylabel('Fooled');


%OutPts(B), highest y_values for x_intervals
bin_size = 0.001;
x_min = min(bn_x); x_max = max(bn_x);
n_bin = int32(floor((x_max-x_min)/bin_size))+1;
x = zeros(n_bin,1); y = zeros(n_bin,1); ct = zeros(n_bin,1);
for i=1:size(bn_x,1)
   k = floor((bn_x(i)-x_min)/bin_size)+1;
   y(k) = max(y(k),bn_y(i));
end
for k=1:n_bin
   x(k) = x_min+double(k-1)*bin_size+bin_size/2;
end
idx = (y>0);
x = x(idx); y = y(idx);
% figure;
% plot(x,y,'.');

%ApproximateCurve(OutPts(B))
fit_param = polyfit(x,y,2);
y_fit = polyval(fit_param,x);

%DecisionBoundary
ds = 0; cnt=0;
for i = 1:size(bn_x,1)
  x1 = bn_x(i); y1 = bn_y(i); y2 = polyval(fit_param,x1);
  if (y1 > y2) 
    my_fun = @(x)((x-x1)^2+(polyval(fit_param,x)-y1)^2);
    [x2, d] = cobyla(my_fun, x1);
    ds = ds+sqrt(d);
    cnt = cnt+1;
  end
end
d_thr = ds/cnt;
hold on;
plot(x,y_fit,'-');
hold on;
plot(x,y_fit+d_thr,'-');
legend({'Infected';'Normal';'Fitted';'Threshold'});

po_dt = zeros(size(po_y));
for i=1:size(po_x,1)
  x1 = po_x(i); y1 = po_y(i); y2 = polyval(fit_param,x1);
  if (y1 > y2)
    my_fun = @(x)((x-x1)^2+(polyval(fit_param,x)-y1)^2);
    [x2, d] = cobyla(my_fun, x1);
    po_dt(i) = sqrt(d);
  end
end
bn_dt = zeros(size(bn_y));
for i=1:size(bn_x,1)
  x1 = bn_x(i); y1 = bn_y(i); y2 = polyval(fit_param,x1);
  if (y1 > y2)
    my_fun = @(x)((x-x1)^2+(polyval(fit_param,x)-y1)^2);
    [x2, d] = cobyla(my_fun, x1);
    bn_dt(i) = sqrt(d);
  end
end
tgt = [zeros(size(bn_y));ones(size(po_y))];
scs = [bn_dt;po_dt];
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
