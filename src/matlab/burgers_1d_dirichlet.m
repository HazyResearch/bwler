% Parameters
nu = 0.01/pi;        % viscosity parameter
dom = [-1 1];        % spatial domain
x = chebfun('x', dom);
t = linspace(0,1,201);  % time domain, matching original 200 steps

% Initial condition
u0 = -sin(pi*x);

% PDE right-hand side
f = @(t, x, u) nu*diff(u,2) - u.*diff(u);

% Solve using pde15s with Dirichlet BCs
u = pde15s(f, t, u0, 'dirichlet');

% Extract solution values on a grid for plotting
nn = 511;  % spatial resolution
x = linspace(-1, 1, nn);
usol = zeros(length(t), nn);

% Evaluate the solution
for i = 1:length(t)
    usol(i,:) = feval(u(:,i), x);
end

% Plot
pcolor(t, x, usol');
shading interp
axis tight
colormap(jet)
xlabel('Time')
ylabel('Space')
title('Burgers equation with Dirichlet BCs')
colorbar

% Save results
save('burgers_1d_dirichlet.mat','t','x','usol')