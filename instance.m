clc
clear 
close all

imageDir = fullfile(toolboxdir('vision'),'visiondata','upToScaleReconstructionImages');
images = imageDatastore(imageDir);
I1 = readimage(images, 1);
I2 = readimage(images, 2);
figure
imshowpair(I1, I2, 'montage'); 
title('Original Images');

% Load precomputed camera intrinsics
data = load('sfmCameraIntrinsics.mat');
intrinsics = data.intrinsics;

I1 = undistortImage(I1, intrinsics);
I2 = undistortImage(I2, intrinsics);
figure 
imshowpair(I1, I2, 'montage');
title('Undistorted Images');

% Detect feature points
imagePoints1 = detectMinEigenFeatures(im2gray(I1), MinQuality = 0.1);

% Visualize detected points
figure
imshow(I1, InitialMagnification = 50);
title('150 Strongest Corners from the First Image');
hold on
plot(selectStrongest(imagePoints1, 150));

% Create the point tracker
tracker = vision.PointTracker(MaxBidirectionalError=1, NumPyramidLevels=5);

% Initialize the point tracker
imagePoints1 = imagePoints1.Location;
initialize(tracker, imagePoints1, I1);

% Track the points
[imagePoints2, validIdx] = step(tracker, I2);
matchedPoints1 = imagePoints1(validIdx, :);
matchedPoints2 = imagePoints2(validIdx, :);

% Visualize correspondences
figure
showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2);
title('Tracked Features');

% Estimate the fundamental matrix
[E, epipolarInliers] = estimateEssentialMatrix(...
    matchedPoints1, matchedPoints2, intrinsics, Confidence = 99.99);

% Find epipolar inliers
inlierPoints1 = matchedPoints1(epipolarInliers, :);
inlierPoints2 = matchedPoints2(epipolarInliers, :);

% Display inlier matches
figure
showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2);
title('Epipolar Inliers');

relPose = estrelpose(E, intrinsics, inlierPoints1, inlierPoints2);

% Detect dense feature points. Use an ROI to exclude points close to the
% image edges.
border = 30;
roi = [border, border, size(I1, 2)- 2*border, size(I1, 1)- 2*border];
imagePoints1 = detectMinEigenFeatures(im2gray(I1), ROI = roi, ...
    MinQuality = 0.001);

% Create the point tracker
tracker = vision.PointTracker(MaxBidirectionalError=1, NumPyramidLevels=5);

% Initialize the point tracker
imagePoints1 = imagePoints1.Location;
initialize(tracker, imagePoints1, I1);

% Track the points
[imagePoints2, validIdx] = step(tracker, I2);
matchedPoints1 = imagePoints1(validIdx, :);
matchedPoints2 = imagePoints2(validIdx, :);

% Compute the camera matrices for each position of the camera
% The first camera is at the origin looking along the Z-axis. Thus, its
% transformation is identity.
% 投影矩阵
camMatrix1 = cameraProjection(intrinsics, rigidtform3d);
camMatrix2 = cameraProjection(intrinsics, pose2extr(relPose));

% Compute the 3-D points
points3D = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);

% Get the color of each reconstructed point
numPixels = size(I1, 1) * size(I1, 2);
allColors = reshape(I1, [numPixels, 3]);
colorIdx = sub2ind([size(I1, 1), size(I1, 2)], round(matchedPoints1(:,2)), ...
    round(matchedPoints1(:, 1)));
color = allColors(colorIdx, :);

% Create the point cloud
ptCloud = pointCloud(points3D, 'Color', color);



% Visualize the camera locations and orientations
cameraSize = 0.3;
figure
plotCamera(Size=cameraSize, Color='r', Label='1', Opacity=0);
hold on
grid on
plotCamera(AbsolutePose=relPose, Size=cameraSize, ...
    Color='b', Label='2', Opacity=0);

% Visualize the point cloud
pcshow(ptCloud, VerticalAxis='y', VerticalAxisDir='down', MarkerSize=45);

% Rotate and zoom the plot
camorbit(0, -30);
camzoom(1.5);

% Label the axes
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis')

title('Up to Scale Reconstruction of the Scene');




% Detect the globe
globe = pcfitsphere(ptCloud, 0.1);

% Display the surface of the globe
plot(globe);
title('Estimated Location and Size of the Globe');
hold off


% Determine the scale factor
scaleFactor = 10 / globe.Radius;

% Scale the point cloud
ptCloud = pointCloud(points3D * scaleFactor, Color=color);
relPose.Translation = relPose.Translation * scaleFactor;

% Visualize the point cloud in centimeters
cameraSize = 2; 
figure
plotCamera(Size=cameraSize, Color='r', Label='1', Opacity=0);
hold on
grid on
plotCamera(AbsolutePose=relPose, Size=cameraSize, ...
    Color='b', Label='2', Opacity=0);

% Visualize the point cloud
pcshow(ptCloud, VerticalAxis='y', VerticalAxisDir='down', MarkerSize=45);
camorbit(0, -30);
camzoom(1.5);

% Label the axes
xlabel('x-axis (cm)');
ylabel('y-axis (cm)');
zlabel('z-axis (cm)')
title('Metric Reconstruction of the Scene');












