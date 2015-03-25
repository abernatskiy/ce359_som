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




