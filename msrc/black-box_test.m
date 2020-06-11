addpath(genpath('msrc'))

%initial trigger
% mu = 0.5;
% sigma = 0.13;
% width = 32;
% pt = zeros(width,width);
% m = 7*7;
% for i=1:m
%     z = -1;
%     while (z < 0) || (1 < z)
%         z = normrnd(mu,sigma);
%     end
%     i = ceil(z*width);
%     z = -1;
%     while (z < 0) || (1 < z)
%         z = normrnd(mu,sigma);
%     end
%     j = ceil(z*width);
%     pt(i,j) = 1;
% end
% imshow(pt);
% imwrite(pt,'trigger_try.png');
%%
%parameter settings
args.sigma = 1e-3;
args.epsilon = 0.05;
args.samples_per_draw = 20;
args.max_queries = 10000;
args.plateau_drop=2.0;
args.plateau_length=10;
args.momentum = 0.9;
args.max_lr=1e-2;
args.min_lr=5e-5;
args.adv_threshold = -1.0;
args.conservative = 2;


load('preload_img');
[sc_record,tg_record,tm_record] = black_box(args,pt);
%%
% for i=1:size(tg_record,1)
%     disp(i);
%     imshow(tg_record{i});
%     pause;
% end

%%


