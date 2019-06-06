function [fnp] = bootstrap_test(y)
  [N,M] = size(y);
  
  n = 10000;
  x = 1:N;
  xx = bootstrp(n,@identity_copy,x);
  fnp = 0;
  for i=1:n
      yy = y(xx(i,:))';
      aa = calc_anomaly_index(yy);
      idx = (log(aa) <= 2) & (xx(i,:)<1);
%       idx = (log(aa) > 2) & (xx(i,:)>1);
      sum(idx);
      fnp = fnp+sum(idx)/N;
  end
  fnp = fnp/n;
end