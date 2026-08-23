function reflectivity = PC_Create_Myelin_Target(N, num_fibers)
%{
Synthetic myelin-fiber-like target on an N x N grid. Complex reflectivity
= amplitude .* exp(1i * phase). Endpoints are generated automatically so
the function can be called non-interactively from the synthetic-data
pipeline (matching PC_Generate_Siemens_Target's style).

Each fiber grows from a random start to a random end with a natural
squiggle (small random angle perturbation per step) and a tapered
amplitude profile. Background is zero (no speckle) so empty regions
stay clean. Amplitude is rescaled to [0, 1] and phase to [-pi, pi].

Inputs:
  N           : grid side (pixels)
  num_fibers  : number of fibers to grow

Output:
  reflectivity : N x N complex array (amp .* exp(1i * phase))
%}

    thickness   = 0.7;
    amp_range   = [0, 1.0];
    phase_range = [-pi, pi];

    % Empty background — fibers are drawn on a clean field.
    amp_map   = zeros(N);
    phase_map = zeros(N);

    % Random fiber endpoints. Minimum fiber length ~0.25 * N to keep
    % fibers visible; endpoints are otherwise uniformly distributed
    % within the grid.
    min_len = 0.25 * N;
    endpoints = zeros(num_fibers, 4);   % [x1 y1 x2 y2] per fiber
    for k = 1:num_fibers
        while true
            pts = 1 + (N - 1) * rand(1, 4);
            if hypot(pts(3) - pts(1), pts(4) - pts(2)) >= min_len, break; end
        end
        endpoints(k, :) = pts;
    end

    [X, Y] = meshgrid(1:N, 1:N);

    for k = 1:num_fibers
        x = endpoints(k, 1); y = endpoints(k, 2);
        x_end = endpoints(k, 3); y_end = endpoints(k, 4);

        dist       = hypot(x_end - x, y_end - y);
        fiber_len  = round(dist);
        angle_var  = atan2(y_end - y, x_end - x);

        % Tapered amplitude profile along the fiber.
        taper_profile = exp(-((1:fiber_len) - fiber_len/2).^2 / (0.6 * fiber_len)^2);

        for step = 1:fiber_len
            % Natural squiggle: update angle with small randomness.
            angle_var = angle_var + 0.1 * randn();

            % Move forward one pixel-step.
            x = x + cos(angle_var);
            y = y + sin(angle_var);

            % Stop if outside grid.
            if x < 1 || x > N || y < 1 || y > N
                break;
            end

            % Gaussian blob at (x, y).
            blob = exp(-((X - x).^2 + (Y - y).^2) / (2 * thickness^2));

            % Per-fiber random amplitude scaling + 4-state random phase.
            min_fiber_amp = 0.4;
            fiber_amp = rand() * (1.0 - min_fiber_amp) + min_fiber_amp;
            a = fiber_amp * taper_profile(step);
            p = pi * randsample([-1, -0.5, 0.5, 1], 1);

            amp_map   = amp_map   + a * blob;
            phase_map = phase_map + p * blob;
        end
    end

    % Normalize and combine.
    amp_map      = rescale(amp_map,   amp_range(1),   amp_range(2));
    phase_map    = rescale(phase_map, phase_range(1), phase_range(2));
    reflectivity = amp_map .* exp(1i * phase_map);
end
