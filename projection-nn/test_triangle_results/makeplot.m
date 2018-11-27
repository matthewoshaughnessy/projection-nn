load triangle_test.mat
clf;
lims = 1.0*[-1 1 -1 1];

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
mask(4,:) = P'*[0;1]  -0.5;% & P'*[1;1] > 0.5;
mask(5,:) = P'*[0;1] < -0.5 & P'*[-1;1] < 0.5 & P'*[1;1] < 0.5;
mask(6,:) = P'*[0;1] < -0.5 & P'*[-1;1] > 0.5;
mask(7,:) = P'*[0;1] > -0.5 & P'*[-1;1] < 0.5 & P'*[1;1] < 0.5;

l2errs = norms(Pproj - Pproj_hat);
region_meanl2errs = zeros(1,7);
for i = 1:7
  region_meanl2errs(i) = mean(l2errs(mask(i,:)));
end

subplot(2,4,1);
plot(P(1,:),P(2,:),'k.'); hold on;
plot(Pproj(1,:),Pproj(2,:),'b.');
plot(Pproj_hat(1,:),Pproj_hat(2,:),'r.');
axis(lims); grid on; axis square;
title('Original data'); set(gca,'fontsize',16);
legend('Ground truth projection','Network projection','Point');
for i = 1:7
  subplot(2,4,i+1);
  plot(P(1,:),P(2,:),'k.','linew',1); hold on;
  plot(P(1,mask(i,:)),P(2,mask(i,:)),'.','linew',1);
  axis(lims); grid on; axis square;
  title({sprintf('Region %d',i), ...
    sprintf('mean l2 err = %.4f',region_meanl2errs(i))});
  set(gca,'fontsize',16);
end
