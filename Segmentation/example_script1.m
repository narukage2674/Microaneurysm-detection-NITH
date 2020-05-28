% Here is an example script for cl_classify function
% provided with DiaRetDB1 toolkit

load traindata traindatahist
f=dir('F:/Healthy/*.jpg');
fnames={f.name};
for a=1:length(fnames)
    img = imread(strcat('F:/Healthy/',fnames{a}));%'../images/ddb1_fundusimages/image019.png'
    result = cl_classify(traindatahist, img, 0.05);
    result = bwareaopen(result,10);
    result = imfill(result,'holes'); 
    imwrite(repmat(result,[1 1 3]).*(double(img)/255),strcat(fnames{a}(1:9),'png'));
end
