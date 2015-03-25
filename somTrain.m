% somTrain.m
% Self-organizing map
% By Anton Bernatskiy, abernats@uvm.edu
% March 22 2015

function kWts = somTrain(xPats, kWts, kIterations)
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
%		radius = ceil(maxRadius*(1+kIterations-iter)/kIterations);
		radius = 1;
		% Ones are added to the sums in oder to not waste the last iteration by having alpha=0 at that time

		% The order in which the patterns are presented is determined by a randomly permuted vector perm
		perm = randperm(nPats);

		for p = 1:nPats
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


