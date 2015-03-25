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





