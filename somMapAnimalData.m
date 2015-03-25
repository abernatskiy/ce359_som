% somMapAnimalData.m
% Self-organizing map
% By Anton Bernatskiy, abernats@uvm.edu
% March 21 2015

% Setting the parameters
kTrainingIterations = 1000;
kGridSide = 20;
noiseAmplitude = 0.1; % determines the maximum deviation of initial Kohonen vectors from centrod of the training set

% Reading the dataset
[xPats, txt, raw] = xlsread('AnimalData.xls');
patLabels = txt(2:end,1);
attrLabels = txt(1,2:end);
% The data is already normalized in L2 sense, so no additional normalization applied

% Finding out the number of input nodes
xSize = size(xPats, 2);

% Initializing weights with random values from [0,1]
%initialKohonenWts = rand(xSize, kGridSide, kGridSide+1); % replace with noisified centroid

% Initializing weights with noisified dataset centroid location
centroid = transpose(mean(xPats, 1));
initialKohonenWts = centroid(:, ones(1, kGridSide), ones(1, kGridSide+1));
% Noisifying...
initialKohonenWts = initialKohonenWts + 2*noiseAmplitude*(rand(size(initialKohonenWts)) - 0.5);
% ...and cutting the values to not go outside of [0,1]
initialKohonenWts = arrayfun(@(x) (x>=1) + x*and(x<1, x>0), initialKohonenWts);

% Training the network
kohonenWts = somTrain(xPats, initialKohonenWts, kTrainingIterations, patLabels);

% Unified distance matrix plot
%somUDMPlot(kohonenWts, xPats, patLabels, strcat('udm', num2str(kGridSide), 'iter', num2str(kTrainingIterations)));

% Debug plots: distances from each training pattern to Kohonen vectors plotted as a heatmap on the grid
if 0 % Commenting out the debug plot
for p = 1:size(xPats,1)
	figure()
	curVec = transpose(xPats(p, :));
	diffs = kohonenWts - curVec(:, ones(1, kGridSide), ones(1, kGridSide));
	patternMap = [];
	patternMap(:,:) = sum(diffs.^2, 1);
	[ X1, X2 ] = meshgrid([1:kGridSide+1], [1:kGridSide+1]);
	colormap(hsv(10));
	graph = pcolor(X1, X2, patternMap);
	set(graph, 'edgecolor', 'none');
	colorbar;
	title(strcat('Pattern ', num2str(p), ' distance'));
end
end % End of comment

