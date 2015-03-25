% somMapAnimalData.m
% Self-organizing map
% By Anton Bernatskiy, abernats@uvm.edu
% March 21 2015

% Setting the parameters
kTrainingIterations = 50000;
kGridSide = 10;
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
initialKohonenWts = centroid(:, ones(1, kGridSide), ones(1, kGridSide));
% Noisifying...
initialKohonenWts = initialKohonenWts + 2*noiseAmplitude*(rand(size(initialKohonenWts)) - 0.5);
% ...and cutting the values to not go outside of [0,1]
initialKohonenWts = arrayfun(@(x) (x>=1) + x*and(x<1, x>0), initialKohonenWts);

% Training the network
kohonenWts = somTrain(xPats, initialKohonenWts, kTrainingIterations);

% Plotting

%%% Unified distance matrix plot
figure()
udm = somComputeUDM(kohonenWts)
colormap(flipud(gray));
%brighten(0.7);
graph = sanePColor([1:kGridSide], [1:kGridSide], udm);
set(graph, 'edgecolor', 'none');
colorbar;
title('Unified distance matrix for animal data');

% Adding labels to see where do training patterns lie
[ labelLocations, labelIndices ] = sortrows(somClosestVectors(kohonenWts, xPats));
line(labelLocations(:,1), labelLocations(:,2), ones(size(labelLocations,1)), 'linestyle', 'none', 'color','r', 'marker', '+', 'markeredgecolor', 'black', 'markerfacecolor', 'k','MarkerSize', 12);
curpoint = labelLocations(1, :);
curline = patLabels(labelIndices(1));
pat = 2;
% The following construction concatenates the labels of identical vectors to make them readable on the plot
while 1
	if pat>size(xPats, 1)
		text(curpoint(1), curpoint(2), curline, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
		break;
	end
	if isequal(curpoint, labelLocations(pat,:))
		curline = strcat(curline, ', ', patLabels(labelIndices(pat)));
	else
		text(curpoint(1), curpoint(2), curline, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
		curpoint = labelLocations(pat,:);
		curline = patLabels(labelIndices(pat));
	end
	pat = pat + 1;
end
% Adjusting axes to make labels readable
axis([0, kGridSide+1, 0, kGridSide+1]);
caxis([0, max(max(udm))*1.2]);
print(strcat('udm', num2str(kGridSide)), '-dpng')

%%% Debug plots: distances from each training pattern to Kohonen vectors plotted as a heatmap on the grid
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

