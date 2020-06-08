home_folder = getenv('HOME');
fo = fullfile(home_folder,'/data/npys');
mat_folder = fullfile(home_folder,'/data/mats/backdoor');

N = 100;
c_Se = cell(N,1);
c_Su = cell(N,1);
for i=1:N
    fn = ['out_',num2str(i-1)];

    [features,labels,ori_labels] = read_features(fn,fo);
    [gb_model] = global_model(features, labels);
    c_Se{i} = gb_model.Se;
    c_Su{i} = gb_model.Su;
end
%%
dists_Se = zeros(N,N);
dists_Su = zeros(N,N);
for i = 1:N
    for j = i+1:N
        dif = c_Se{i}-c_Se{j};
        dists_Se(i,j) = norm(dif);
        dif = c_Su{i}-c_Su{j};
        dists_Su(i,j) = norm(dif);
    end
end
norm_Se = zeros(N,1);
norm_Su = zeros(N,1);
for i = 1:N
    norm_Su(i) = norm(c_Su{i});
    norm_Se(i) = norm(c_Se{i});
end
%%
save('dist_Ses','dists_Se','dists_Su','c_Se','c_Su');
%%
l_e = zeros(2,1);
l_u = zeros(2,1);
k = 0;
for i=1:N
    for j =i+1:N
        k = k+1;
        l_e(k) = dists_Se(i,j);
        l_u(k) = dists_Su(i,j);
    end
end
%%
[y,x] = hist(norm_Su,100);
tt_y = sum(y);
for i=2:size(y,2)
    y(i) = y(i)+y(i-1);
end
h1 = plot(x,100*y/tt_y,'--');
[y,x] = hist(l_u,100);
tt_y = sum(y);
for i=2:size(y,2)
    y(i) = y(i)+y(i-1);
end
hold on;
h2 = plot(x,100*y/tt_y);
legend([h1,h2],{'Original','Distance'});
xlabel('Norm size');
ylabel('% of data');
set(gcf,'Position',[100 100 260 200])
%%
[y,x] = hist(norm_Se,100);
tt_y = sum(y);
for i=2:size(y,2)
    y(i) = y(i)+y(i-1);
end
h1 = plot(x,y/tt_y,'--');

[y,x] = hist(l_e,100);
tt_y = sum(y);
for i=2:size(y,2)
    y(i) = y(i)+y(i-1);
end
hold on;
h2 = plot(x,y/tt_y);
legend([h1,h2],{'Original','Distance'});
xlabel('Norm');
ylabel('% of data');
set(gcf,'Position',[100 100 260 200])