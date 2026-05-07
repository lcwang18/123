% This is a simplest implementation of the proposed RIFT algorithm. In this implementation,...
% rotation invariance part and corner point detection are not included.

clc;clear;close all;
warning('off')

addpath sar-optical   % type of multi-modal data

str1='sar-optical/pair1.jpg';   % image pair
str2='sar-optical/pair2.jpg';

%str1="D:\sourcecode\ImageRegistration\Image-Registration-master\test images\PB-1.jpg";
%str2="D:\sourcecode\ImageRegistration\Image-Registration-master\test images\PB-2.jpg";

%str1="D:\compare img\LiDARintensity_ref.tif";
%str2="D:\compare img\optical_sen.tif";

%str1="D:\sourcecode\ImageRegistration\Image-Registration-master\test images\SAR-SIFT_2.JPG"
%str2="D:\sourcecode\ImageRegistration\Image-Registration-master\test images\SAR-SIFT_1.JPG"

%str1="D:\compare img\Optical_ref2.tif"
%str2="D:\compare img\SAR_sen2.tif"

str2="D:\dataset\Gaza_512\train\ref\opt (59).tif";   % image pair
str1="D:\dataset\Gaza_512_trans\train1\sen\sar59.png";

im1 = im2uint8(imread(str1));
im2 = im2uint8(imread(str2));

if size(im1,3)==1
    temp=im1;
    im1(:,:,1)=temp;
    im1(:,:,2)=temp;
    im1(:,:,3)=temp;
end

if size(im2,3)==1
    temp=im2;
    im2(:,:,1)=temp;
    im2(:,:,2)=temp;
    im2(:,:,3)=temp;
end

disp('RIFT feature detection and description')
% RIFT feature detection and description
tic
[des_m1,des_m2] = RIFT_no_rotation_invariance(im1,im2,4,6,96);

disp('nearest matching')
% nearest matching
[indexPairs,matchmetric] = matchFeatures(des_m1.des,des_m2.des,'MaxRatio',1,'MatchThreshold', 100);
matchedPoints1 = des_m1.kps(indexPairs(:, 1), :);
matchedPoints2 = des_m2.kps(indexPairs(:, 2), :);
[matchedPoints2,IA]=unique(matchedPoints2,'rows');
matchedPoints1=matchedPoints1(IA,:);

disp('outlier removal')
%outlier removal
H=FSC(matchedPoints1,matchedPoints2,'affine',2);
Y_=H*[matchedPoints1';ones(1,size(matchedPoints1,1))];
Y_(1,:)=Y_(1,:)./Y_(3,:);
Y_(2,:)=Y_(2,:)./Y_(3,:);
E=sqrt(sum((Y_(1:2,:)-matchedPoints2').^2));
inliersIndex=E<3;
cleanedPoints1 = matchedPoints1(inliersIndex, :);
cleanedPoints2 = matchedPoints2(inliersIndex, :);
fprintf('the total matching time is %f\n',toc);

disp('Show matches')
% Show results
figure; showMatchedFeatures(im1, im2, cleanedPoints1, cleanedPoints2, 'montage');

disp('registration result')
% registration
image_fusion(im2,im1,double(H));
