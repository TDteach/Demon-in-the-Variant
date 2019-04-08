function [index] = calc_anomaly_index(a)
%CALC_ANOMALY_INDEX Summary of this function goes here
%   Detailed explanation goes here
  mm = median(a);
  b = abs(a-mm);
  mm = median(b)*1.4826;
  index = b/mm;
end

