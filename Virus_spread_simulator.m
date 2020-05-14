% Suscept + Infected -k1-> 2 Infected
% Infected -k2-> Recovered
% Infected -k3-> Dead

dates = 1:60; % 100 days
timestep = 0.01; % Simulate every 0.001 day
sim_timestep = 1:round(dates(end)/timestep);
SIRD = zeros(5,length(sim_timestep));
dSIRD = SIRD;
Population = 3e8;
quarantineRatio = 0.99
SIRD(:,1) = [Population * (1-quarantineRatio), ...
             70000, 0, 0, ...
             Population * quarantineRatio];
k_SI = 60e-8;
k_IR = 1.5

k_SI = k_SI;
k_IR = k_IR;
k_ID = k_IR / 100;
k_SQ = 0.0001; 
k_QS = 0.0001;
k_DS = 23;

for i = 2:sim_timestep(end)
    k_SQ = k_SQ * power((1 + dSIRD(4,i-1) / k_DS),timestep);
    dSIRD(:,i) = [SIRD(1,i-1) * SIRD(2,i-1) * -k_SI - k_SQ * SIRD(1,i-1) + k_QS * SIRD(5,i-1), ...
                  SIRD(1,i-1) * SIRD(2,i-1) * k_SI - SIRD(2,i-1) * (k_IR + k_ID), ...
                  SIRD(2,i-1) * k_IR, ...
                  SIRD(2,i-1) * k_ID, ...
                  k_SQ * SIRD(1,i-1) - k_QS * SIRD(5,i-1)]' * timestep;
    SIRD(:,i) = SIRD(:,i-1) + dSIRD(:,i);
end

SIRD(:,end)

trapz(timestep,sum(dSIRD(2:4,:),1))

figure(303); clf; hold on
subplot(211)
grid on; hold on;
plot(sim_timestep * timestep, SIRD(2,:))
yyaxis right
plot(sim_timestep * timestep, SIRD(3,:))
plot(sim_timestep * timestep, SIRD(4,:)*10,'-x')
subplot(212);
plot(sim_timestep * timestep, dSIRD(2:4,:))