function [index] = calc_anomaly_index(a)
%CALC_ANOMALY_INDEX Summary of this function goes here
%   Detailed explanation goes here
  ma = median(a);
  b = abs(a-ma);
  mm = median(b)*1.4826;
%   index = max(0, a-ma)/mm;
  index = b/mm;
end

