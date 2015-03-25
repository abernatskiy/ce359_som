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


