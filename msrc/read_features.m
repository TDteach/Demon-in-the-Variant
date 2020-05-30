function [ft,lb,o_lb] = read_features(fn,fo)
%READ_FEATURES Summary of this function goes here
%   Detailed explanation goes here
if nargin<2
  fo = fullfile(getenv('HOME'),'/data/npys/');
end

ft = readNPY(fullfile(fo,[fn,'_X.npy']));
lb = readNPY(fullfile(fo,[fn,'_labels.npy']));
o_lb = readNPY(fullfile(fo,[fn,'_ori_labels.npy']));
n = size(ft,1);
lb = lb(1:n);
o_lb = o_lb(1:n);
end

