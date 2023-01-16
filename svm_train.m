% svm_train

up=[imagesdemobottrain;imagesdemobotval];
bot=[imagesdemotoptrain;imagesdemotopval];
% up=[imagesup23train];
% bot=[imagesbot23train];
save_svm_path='svm_demo';
test_X=[imagesdemotoptest(:,2),imagesdemobottest(:,2)];
test_Y=imagesdemotoptest(:,1);

X=[up(:,2),bot(:,2)];
Y=up(:,1);

svm_model=fitcsvm(X,Y,'KernelFunction','linear');

[label,score]=predict(svm_model,test_X);
ind=label==test_Y;
sum(ind)/length(ind)
save(save_svm_path,"svm_model")
