% 欧式几何恢复
% 通过单目两张图像进行同名点位置重建

clc
clear 
close all

img1 = imread("D:\Program Files\MATLAB\R2022b\toolbox\vision\visiondata\structureFromMotion\image1.jpg");
img2 = imread("D:\Program Files\MATLAB\R2022b\toolbox\vision\visiondata\structureFromMotion\image2.jpg");
data = load("D:\Program Files\MATLAB\R2022b\toolbox\vision\visiondata\structureFromMotion\cameraParams.mat");

cameraParams = data.cameraParams;

%获得内参矩阵
intrinsics = cameraParams.Intrinsics;
K = intrinsics.K;
%对第一张图像进行去畸变校正
img1 = undistortImage(img1, intrinsics); 
img2 = undistortImage(img2, intrinsics); 

img1 = im2gray(img1);
img2 = im2gray(img2);

figure
subplot(121),imshow(img1);
subplot(122),imshow(img2);

border = 50;
roi = [border, border, size(img1, 2)- 2*border, size(img1, 1)- 2*border];
points1 = detectSURFFeatures(img1, NumOctaves=8, ROI=roi);
points2 = detectSURFFeatures(img2, NumOctaves=8, ROI=roi);

figure
subplot(121),
imshow(img1); hold on;
plot(points1.selectStrongest(100));hold off;

subplot(122),
imshow(img2); hold on;
plot(points2.selectStrongest(100));hold off;

[features1, points1] = extractFeatures(img1, points1);
[features2, points2] = extractFeatures(img2, points2);

indexPairs   = matchFeatures(features1, features2,MaxRatio=0.7, Unique=true);

% 同名点
matchedPoints1 = points1(indexPairs(:, 1));
matchedPoints2 = points2(indexPairs(:, 2));
[F,ispoint] = estimateFundamentalMatrix(matchedPoints1,matchedPoints2,'Method','Norm8Point');
% 获得本质矩阵
E = K'*F*K;
% 本质矩阵分解获得RT
[U,~,V] = svd(E);
W = [0,-1,0;1,0,0;0,0,1];
R1 = U*W*V;
R2 = U*W'*V;

% 选择最合适的解
if det(R1) < 0
    R1 = -R1;
end
if det(R2) < 0
    R2 = -R2;
end

% 计算平移向量
Z = [0 1 0; -1 0 0 ; 0 0 0];
A = U*Z*U';
[AU,AS,AV] = svd(A);

T1 = AV(:,3);
T2 = -AV(:,3);
%四个可能解
pose1 = [R1,T1];
pose2 = [R1,T2];
pose3 = [R2,T1];
pose4 = [R2,T2];

% 通过三角化确定唯一正确的一组解
u1 = 290.34955;
v1 = 143.91621;

u2 = 278.50464;	
v2 = 147.92189;

RT1 = [1 0 0 0;0 1 0 0;0 0 1 0];

M1 = K * RT1;
M2 = K * pose2;
% M22 = K * pose2;
% M23 = K * pose3;
% M24 = K * pose4;

m11 = M1(1,:);
m12 = M1(2,:);
m13 = M1(3,:);

m21 = M2(1,:);
m22 = M2(2,:);
m23 = M2(3,:);

Amat = [u1*m13-m11;v1*m13-m12;u2*m23-m21;v2*m23-m22];

[~,~,Vvvv] = svd(Amat);

homogeneous_coords = Vvvv(:, end);

nonhomogeneous_coords = homogeneous_coords(1:3) / homogeneous_coords(4);

nonhomogeneous_coords = [nonhomogeneous_coords;1];
% 正确的RT可以使重建后的点在两个相机的相机坐标系下均具有正深度
p1 = RT1 * nonhomogeneous_coords;
p2 = pose2 * nonhomogeneous_coords;

disp("p1的值:")
disp(p1)

disp("p2的值:")
disp(p2)






