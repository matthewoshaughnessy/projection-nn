fileprefix = 'test-capacity-';
filepostfix = '_test.mat';

testnames = dir([fileprefix '*' filepostfix]);
testnames = {testnames.name};
expids = cellfun(@(x)strtok(x,'_'),testnames,'uniformoutput',false);
expids = cellfun(@(x)x(length(fileprefix)+1:end),expids,'uniformoutput',false);
expids = cellfun(@(x)str2num(strrep(x,'-',' ')),expids,'uniformoutput',false);
expids = cat(1,expids{:});
layersize1 = expids(:,1);
layersize2 = expids(:,2);
randseeds = expids(:,3);
s1 = sort(unique(layersize1));  ns1 = length(s1);
s2 = sort(unique(layersize2));  ns2 = length(s2);
r = sort(unique(randseeds));    nr = length(r);

mean_errs = zeros(ns1,ns2,nr);
for is1 = 1:length(unique(layersize1))
  for is2 = 1:length(unique(layersize2))
    for ir = 1:nr
      filename = sprintf('%s%d-%d-%d%s',fileprefix,s1(is1),s2(is2),r(ir),filepostfix);
      fprintf('Parsing file %s...', filename);
      load(filename);
      mean_errs(is1,is2,ir) = mean(errs);
      fprintf('\n');
    end
  end
end

save('mean_errs','mean_errs','s1','s2','r');
%clf;
%imagesc(s1,s2,mean(mean_errs,3).');
%set(gca,'xtick',s1,'ytick',s2,'ydir','normal','fontsize',16);

% for i = 1:length(testnames)
%   load(testnames{i});
%   [ir,is] = ind2sub([nr,ns],i);
%   subplotind = sub2ind([ns,nr],is,ir);
%   subplot(length(randseeds),length(netsizes),subplotind);
%   plot(Pproj(1,:),Pproj(2,:),'k.','markersize',0.1); hold on;
%   plot(Pproj_hat(1,:),Pproj_hat(2,:),'b.','markersize',0.1);
%   axis([-0.6 0.6 -0.6 0.6]); axis square; grid on;
%   set(gca,'xtick',-.6:.3:.6,'ytick',-.6:.2:.6,'xticklabel',[],'yticklabel',[]);
%   %title(strrep(testnames{i},'_','\_'));
%   xlabel(sprintf('err = %.2e',mean(errs)));
%   if is == 1, ylabel(sprintf('randseed %s',randseeds{ir})); end
%   if ir == 1, title(sprintf('size %s',netsizes{is})); end
%   set(gca,'fontsize',16);
% end
