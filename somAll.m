% somMapAnimalData.m
% Self-organizing map
% By Anton Bernatskiy, abernats@uvm.edu
% March 21 2015

% Setting the parameters
kTrainingIterations = 10000;
kGridSide = 3;
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
kohonenWts = somTrain(xPats, initialKohonenWts, kTrainingIterations, patLabels);

% Unified distance matrix plot
somUDMPlot(kohonenWts, xPats, patLabels, strcat('udm', num2str(kGridSide), 'x', num2str(kGridSide), '_iter', num2str(kTrainingIterations)));

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


% somTrain.m
% Self-organizing map
% By Anton Bernatskiy, abernats@uvm.edu
% March 22 2015

function kWts = somTrain(xPats, kWts, kIterations, patLabels)
	% Finding out the number of input nodes and Kohonen grid dimensions
	sizeX = size(xPats, 2);
	if sizeX ~= size(kWts, 1)
    error('som:sizeCheck', 'somTrain: size of input training patterns differs from the size of the supplied initial Kohonen vectors, exiting')
  end
	sizeK1 = size(kWts, 2);
	sizeK2 = size(kWts, 3);

	% Finding out the numbers of training patterns
	nPats = size(xPats, 1);

	% Initializing a convergence table
	% The table has one column per each training pattern
	% Each column holds time series of the minimal cartesian distance between the training pattern and the closest Kohonen vector
	convTable = zeros(kIterations, nPats); % COMMENT OUT FOR MAX PERFORMENCE

	% Setting the initial values of training parameters
	% They are both decreasing in the process of training, hence max in their names
	maxAlpha = 0.9;
	% We want neighborhoods to be as large as possible at the beginning, but we can't have the entire grid be the neighborhood,
	% as it would make self-organization process biased towards whichever lucky pattern got to be the last in the first
	% training iteration. It would also waste the first iteration, since no actual self-organization can happen when all Kohonen
  % vectors are adjuxted simultaneously. So, we start with the neighborhoods which cover roughly a quarter to a half of the grid.
	maxRadius = ceil(max(sizeK1, sizeK2)/4);

	% Training the Kohonen layer:
	for iter = 1:kIterations

		% Determining the current values of the learning rule parameters
		alpha = maxAlpha*(1+kIterations-iter)/kIterations;
		radius = ceil(maxRadius*(1+kIterations-iter)/kIterations);
		% Ones are added to the sums in oder to not waste the last iteration by having alpha=0 at that time

		% The order in which the patterns are presented is determined by a randomly permuted vector perm
		perm = randperm(nPats);

		% Plotting the UDM
%		if mod(iter, 100) == 1
%			filename = strcat('udm', num2str(sizeK1), 'x', num2str(sizeK2), '_iter', sprintf('%06d', iter))
%			somUDMPlot(kWts, xPats, patLabels, filename);
%		end

		for p = 1:nPats
			% Plotting the UDM
%			filename = strcat('udm', num2str(sizeK1), 'x', num2str(sizeK2), '_iter', sprintf('%06d', iter), '_pat', sprintf('%02d', p))
%			somUDMPlot(kWts, xPats, patLabels, filename);

			% Finding the differences between each Kohonen vector and the current pattern
			curVec = transpose(xPats(perm(p), :));
			diffs = kWts - curVec(:, ones(1, sizeK1), ones(1, sizeK2));

			% Computing the square of the length of each difference vector and finding minimum
			squareDiffs = diffs.^2;
			net = sum(squareDiffs, 1);
			[ mins1, idxs1 ] = min(net, [], 2);	% could have used ind2sub+sub2ind here, but then it would be more difficult to work with 3D arrays
			[ netmin, idx2 ] = min(mins1, [], 3);
			idx1 = idxs1(idx2);
			% No additional tiebreaking needed due to the properties of the function min

			% Getting list of neighbors
			neighborhood = somNeighborhood(sizeK1, sizeK2, idx1, idx2, radius);
			nNeighbors = size(neighborhood, 1);

			% Adjusting the weights to make the neighborhood of the winning Kohonen vector closer to the current pattern
			diffVec = curVec - kWts(:, idx1, idx2);

			for neighbor = 1:nNeighbors
				idx1 = neighborhood(neighbor, 1);
				idx2 = neighborhood(neighbor, 2);
				kWts(:, idx1, idx2) = kWts(:, idx1, idx2) + alpha*(curVec - kWts(:, idx1, idx2));
			end

			% Storing the new distance between the winning Kohonen vector and the current pattern in the convergence table
			convTable(iter, perm(p)) = sqrt(transpose(diffVec)*diffVec); % COMMENT OUT FOR MAX PERFORMANCE
		end
	end

	% Displaying the convergence time series
	disp(sprintf('\nTime series of the distance between the training patterns and\nclosest Kohonen vectors during the training process:\n'))
	disp(convTable) % COMMENT OUT FOR MAX PERFORMANCE
end


% somNeighborhood.m
% Self-organizing map
% By Anton Bernatskiy, abernats@uvm.edu
% March 21 2015

function indices = somNeighborhood(latsize1, latsize2, coord1, coord2, radius)
	% Computes a list of neighbor nodes on a 2D grid, given dimensions of the grid,
	% coordinates of the center node and the radius of the neighborhood.

	% The return value is an Nx2 matrix, where N is a size of the neighborhood.

	% In the current version the neighborhood is assumed to be a square.
	% The topology of the grid is assumed to be that of a rectangle with boundary.

	% Generating arrays of indices which would be possible if the grid was infinite
	coords1 = [coord1-radius : coord1+radius];
	coords2 = [coord2-radius : coord2+radius];

	% Filtering the arrays to remove indices impossible on a real finite-sized grid
	coords1 = coords1(coords1 > 0);
	coords2 = coords2(coords2 > 0);
	coords1 = coords1(coords1 <= latsize1);
	coords2 = coords2(coords2 <= latsize2);

	% Computing a direct product between the two lists to generate a list of index pairs
	[ grid1, grid2 ] = meshgrid(coords1, coords2);
	indices = [grid1(:), grid2(:)];
end


% somUDMPlot.m
% Self-organizing map
% By Anton Bernatskiy, abernats@uvm.edu
% March 25 2015

function somUDMPlot(kohonenWts, xPats, patLabels, filename)
	% Unified distance matrix plot

	% Determining grid sizes
	sizeK1 = size(kohonenWts, 2);
	sizeK2 = size(kohonenWts, 3);

	% Computing the UDM
	udm = somComputeUDM(kohonenWts);

	% Plotting
	figure('visible', 'off')
	colormap(flipud(gray));
	graph = sanePColor(transpose(udm)); % fixed implementation of pcolor from matlabcentral
	set(graph, 'edgecolor', 'none');
	colorbar;
	title('Unified distance matrix for animal data');

	% Adding labels and markers to see where the training patterns lie
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
	axis([0, sizeK1+1, 0, sizeK2+1]);
	caxis([0, max(max(udm))*1.2]);
	print(filename, '-dpng')

	close all hidden
end


% somComputeUDM.m
% Self-organizing map
% By Anton Bernatskiy, abernats@uvm.edu
% March 22 2015

function udm = somComputeUDM(kWts)
	% Computes unified distance matrix for a set of Kohonen weights

	% Determining dimensions of Kohonen grid and initializing UDM
	sizeK1 = size(kWts, 2);
	sizeK2 = size(kWts, 3);
	udm = zeros(sizeK1, sizeK2);

	for i = 1:sizeK1
		for j = 1:sizeK2
			% Finding the list of adjacent nodes
			adjNodes = [];
			if i>1      adjNodes = [adjNodes; [i-1, j]]; end
			if i<sizeK1 adjNodes = [adjNodes; [i+1, j]]; end
			if j>1      adjNodes = [adjNodes; [i, j-1]]; end
			if j<sizeK2 adjNodes = [adjNodes; [i, j+1]]; end
			numAdjNodes = size(adjNodes, 1);

			% Computing the average distance
			dist = 0;
			for node = 1:numAdjNodes
				% The next line may look scary, but all it does is increment the dist variable by Euclidean distance between the current node and one of its adjacent nodes
				dist = dist + sqrt(sum((kWts(:,i,j) - kWts(:,adjNodes(node,1),adjNodes(node,2))).^2));
				udm(i,j) = dist/numAdjNodes;
			end
		end
	end
end


% somClosestVectors.m
% Self-organizing map
% By Anton Bernatskiy, abernats@uvm.edu
% March 22 2015

function indices = somClosestVectors(kWts, vectors)
	% Given an arbitrary set of vectors and a grid of Kohonen weights,
	% produces a list of positions of closest Kohonen vectors.

	% Determining the number of vectors and Kohonen grid dimensions
	nVecs = size(vectors, 1);
	sizeK1 = size(kWts, 2);
	sizeK2 = size(kWts, 3);

	% Initializing the output array
	indices = zeros(nVecs, 2);

	for v = 1:nVecs
		% For every vector we find the difference between it and all Kohonen vectors...
		curVec = transpose(vectors(v, :));
		diffs = kWts - curVec(:, ones(1, sizeK1), ones(1, sizeK2));
		% ...then its squared magnitude...
		squareDiffs = diffs.^2;
		net = sum(squareDiffs, 1);
		% ... and finally the indices of the vector with the least square distance
		[ mins1, idxs1 ] = min(net, [], 2); % could have used ind2sub+sub2ind here, but then it would be more difficult to work with 3D arrays
		[ netmin, idx2 ] = min(mins1, [], 3);
		indices(v,:) = [ idxs1(idx2), idx2 ];
	end
end


