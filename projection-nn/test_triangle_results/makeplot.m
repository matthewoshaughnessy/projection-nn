load triangle_test.mat
clf;
lims = 2.0*[-1 1 -1 1];

if false
  plot(P(1,:),P(2,:),'k.'); hold on;
  plot(Pproj(1,:),Pproj(2,:),'b.');
  plot(Pproj_hat(1,:),Pproj_hat(2,:),'r.');
  axis(lims); grid on; axis equal;
end

mask = false(7,size(P,2));
mask(1,:) = P'*[0;1] > -0.5 & P'*[-1;1] > 0.5 & P'*[1;1] < 0.5;
mask(2,:) = P'*[-1;1] > 0.5 & P'*[1;1] > 0.5;
mask(3,:) = P'*[0;1] > -0.5 & P'*[-1;1] < 0.5 & P'*[1;1] > 0.5;
mask(4,:) = P'*[0;1] < -0.5 & P'*[1;1] > 0.5;
mask(5,:) = P'*[0;1] < -0.5 & P'*[-1;1] < 0.5 & P'*[1;1] < 0.5;
mask(6,:) = P'*[0;1] < -0.5 & P'*[-1;1] > 0.5;
mask(7,:) = P'*[0;1] > -0.5 & P'*[-1;1] < 0.5 & P'*[1;1] < 0.5;
subsamplemask = rand(1,size(P,2)) < 0.05;

l2errs = norms(Pproj - Pproj_hat);
region_meanl2errs = zeros(1,7);
for i = 1:7
  region_meanl2errs(i) = mean(l2errs(mask(i,:)));
end

subplot(2,4,1);
plot(P(1,subsamplemask),P(2,subsamplemask),'k.'); hold on;
plot(Pproj(1,subsamplemask),Pproj(2,subsamplemask),'b.');
plot(Pproj_hat(1,subsamplemask),Pproj_hat(2,subsamplemask),'r.');
axis(lims); grid on; axis square;
title('Original data'); set(gca,'fontsize',16);
legend('Point','Ground truth projection','Network projection');
for i = 1:7
  subplot(2,4,i+1);
  plot(P(1,subsamplemask),P(2,subsamplemask),'k.','linew',1); hold on;
  plot(P(1,subsamplemask&mask(i,:)),P(2,subsamplemask&mask(i,:)),'.','linew',1);
  axis(lims); grid on; axis square;
  title({sprintf('Region %d',i), ...
    sprintf('mean l2 err = %.4f',region_meanl2errs(i))});
  set(gca,'fontsize',16);
end

exportPlots = false;
if exportPlots
  fprintf('Exporting...');
  %fprintf('pdf...'); export_fig plot.pdf
  fprintf('png...'); export_fig plot.png
  fprintf('done!\n');
end
