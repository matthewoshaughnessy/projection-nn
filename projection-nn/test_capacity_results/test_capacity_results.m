testnames = dir('test_capacity_*_test.mat');
testnames = {testnames.name};
netsizes = sort(unique(cellfun(@(x)x(15:18),testnames,'uniformoutput',false)));
randseeds = sort(unique(cellfun(@(x)x(20),testnames,'uniformoutput',false)));
ns = length(netsizes);
nr = length(randseeds);

cols = cbrewer('qual','Set1',7);

clf;
for i = 1:length(testnames)
  load(testnames{i});
  [ir,is] = ind2sub([nr,ns],i);
  subplotind = sub2ind([ns,nr],is,ir);
  subplot(length(randseeds),length(netsizes),subplotind);
  plot(Pproj(1,:),Pproj(2,:),'k.','markersize',0.1); hold on;
  plot(Pproj_hat(1,:),Pproj_hat(2,:),'b.','markersize',0.1);
  axis([-0.6 0.6 -0.6 0.6]); axis square; grid on;
  set(gca,'xtick',-.6:.3:.6,'ytick',-.6:.2:.6,'xticklabel',[],'yticklabel',[]);
  %title(strrep(testnames{i},'_','\_'));
  xlabel(sprintf('err = %.2e',mean(errs)));
  if is == 1, ylabel(sprintf('randseed %s',randseeds{ir})); end
  if ir == 1, title(sprintf('size %s',netsizes{is})); end
  set(gca,'fontsize',16);
end
