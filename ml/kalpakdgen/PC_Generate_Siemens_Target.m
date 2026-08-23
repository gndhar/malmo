function TARGET = PC_Generate_Siemens_Target(n_spokes, radius_fraction, n_pixels, contrast)
%{
Generates a Siemens star target image.
Inputs:
  n_spokes        : number of spoke pairs
  radius_fraction : spoke length as fraction of image half-width (0 to 1)
  n_pixels        : image size in pixels (default: 373)
  contrast        : Michelson contrast of spokes vs gaps in [0, 1]
                    (default: 1). Mean stays at 0.5 inside the disk so
                    spokes = (1+c)/2, gaps = (1-c)/2. Outside the disk
                    sits at the same gap level so the background is
                    uniform with the gaps.
Output:
  TARGET: n_pixels x n_pixels double matrix representing the Siemens star image
%}
    if nargin < 3, n_pixels = 373; end
    if nargin < 4, contrast = 1;   end

    dtheta = 360 / (2*n_spokes);
    Nc = (n_pixels+1) / 2;
    R = radius_fraction * (n_pixels+1) / 2;

    [x, y] = meshgrid(1:n_pixels);
    theta = mod(atan2d(y-Nc, x-Nc) + dtheta, 360);

    mask = double(mod(fix(theta/dtheta), 2));   % 0/1 spokes
    high = 0.5 * (1 + contrast);
    low  = 0.5 * (1 - contrast);
    TARGET = low + (high - low) * mask;          % spokes=high, gaps=low

    TARGET(((x-Nc).^2 + (y-Nc).^2) > R^2) = low; % outside disk = same as gaps
end
