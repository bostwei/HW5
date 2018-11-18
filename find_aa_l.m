function [dec_zh_aa,dec_zl_aa,dec_zh_l,dec_zl_l,v1_zh_l,v1_zl_l] = find_aa_l(a,j)
%FIND_AA_L This function calculate the choice of aa and l given a for
%working agent
% Input
%   - a is the a given, scaler
%   - j is the age of the worker

% Output 
%   - 
global e r l aa w ttheta ggama bbeta ssigma;
global v0_w_zh v0_w_zl;
    % ------------ the labor l and future asset aaa choice of worker --------------------
    ai = a;
    
    % consumption of worker
    c_w_zh = w * (1-ttheta)* e(1,j)* l + (1+r) * ai - aa';
    c_w_zl = w * (1-ttheta)* e(2,j)* l + (1+r) * ai - aa';
    c_w_zh(find(c_w_zh <0)) = NaN; 
    c_w_zl(find(c_w_zl <0)) = NaN;
  % utility of the worker is 
    u_w_zh = (c_w_zh.^ggama.*(1-l).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zl = (c_w_zl.^ggama.*(1-l).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zh(find(isnan(u_w_zh))) = -inf;
    u_w_zl(find(isnan(u_w_zl))) = -inf;
    
    w_zh = u_w_zh + bbeta * v0_w_zh(:,j+1)';
    w_zl = u_w_zl + bbeta * v0_w_zl(:,j+1)';
   
   % --------------make choice of (l,aa) given a ------------------
   % we frist choose aa
   % - v1_zh_aa the optimal value w_zh after choosing aa, given a. The row of
   % w_zh is varies of l.
   % - dec_zh_aa is the optimal choice of aa given varies of l
   [v1_zh_aa, dec_zh_aa] = max(w_zh,[],2);
   [v1_zl_aa, dec_zl_aa] = max(w_zl,[],2); 
   
   
   % then choose l
   % - v1_zh_aa the optimal value w_zh after choosing aa, given a. The row of
   % w_zh is varies of l.
   % - dec_zh_aa is the optimal choice of aa given varies of l   
   [v1_zh_l,dec_zh_l] = max(v1_zh_aa);
   [v1_zl_l,dec_zl_l] = max(v1_zl_aa);
end

