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
	graph = sanePColor(transpose(udm));
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
