%% Initialisation
format compact;
% close all;
clear all;
clc;

ROS_ENABLE = 1;
if ROS_ENABLE == 1
    rosshutdown
    rosinit         % Initialise rosnode

    % Subscribe/Publish
    sub_traj = rossubscriber('traj', 'trajectory_msgs/JointTrajectory', 'DataFormat', 'struct'); 
    pub = rospublisher('odom','nav_msgs/Odometry');
end

%% Generate MPC using ACADO
EXPORT = 0; % Before running the function 'genAutoraceMPC', change the
            % variable 'srcAcado' to the path of the installed ACADO in the
            % respective computer.

Ts = 0.1;   % Sampling time
N = 5;     % Prediction horizon [steps]

[ carStates, carInputs, carOde, carParams ] = genAutoraceMPC( N, Ts, EXPORT );

%% Simulation - Initialisation
sysStates  = {'$s_x$', '$s_y$', '$\phi$', '$v$'}; % x-COM, y-COM, inertial heading, vehicle speed
sysInputs  = {'$\delta_f$', '$a$'};               % front steering angle, acceleration
sysOutputs = {'$s_x$', '$s_y$', '$\phi$', '$v$'}; % x-COM, y-COM, inertial heading, vehicle speed

n = length(carStates);
m = length(carInputs);
n_U = m;

R = 50;     % Track radius
[xTrack,yTrack] = genTrackRef(R,1000);      % Circular track
xTrack(1) = []; yTrack(1) = [];
xTrack1 = xTrack; yTrack1 = yTrack;

% xTrack = mapTrack(:,1);                     % Recorded map
% yTrack = mapTrack(:,2)-mapTrack(1,2);

% [xTrack,yTrack] = genSineCurve();           % Sine curve

% xTrack = ones(1000,1);                      % Straight line
% yTrack = 1.*linspace(1,1000,1000)';

Tf = 900*.5;
X0 = [xTrack(1), -1, deg2rad(90), 0];   % sx, sy, phi, v    % Init conditions
X0 = [0,        0, deg2rad(0.01), 0];   % sx, sy, phi, v    % Init conditions

input.od = [];

Uref = zeros(N,n_U);
input.u = Uref;

input.W = diag([10,10,0.5,0,10,0.5]); % sx, sy, phi, v, delta_f, a
input.WN = diag([1,1,0.1,0]);

input.W = diag([10,10,10,0,10,0.5]); % sx, sy, phi, v, delta_f, a
input.WN = diag([1,1,1,0]);

% input.W = diag([5,5,1,1,2,0.5]); % sx, sy, phi, v, delta_f, a % recorded track gain
% input.WN = diag([1,1,1,0]);

% input.W = diag([10,10,5,1,2,0.5]); % sx, sy, phi, v, delta_f, a % recorded track gain very good but problem on u-turn
% input.WN = diag([2,2,1,0]);

% input.W = diag([1,1,0,0,10,0.5]); % sx, sy, phi, v, delta_f, a
% input.WN = diag([10,10,0,0]);

%% Simulation - Running
disp('------------------------------------------------------------------');
disp('               Simulation Loop'                                    );
disp('------------------------------------------------------------------');

k = 1;
iter = 0; time = 0;
KKT_MPC = []; INFO_MPC = [];
controls_MPC = [];
state_sim = X0;
input_sim = [];

% visualize_learn;
while time(end) < Tf
%     tic

    if ROS_ENABLE == 1
        % Get the ref for this step and wait for next high level command
        traj1 = receive(sub_traj,70);
        
        % Consume the subscriber data
        x_ilqrRef = GetTraj(traj1);
        xy_ref_traj = x_ilqrRef(1:2,10:21)';
        phi_ref_traj = x_ilqrRef(3,10:21)';
        v_ref_traj = x_ilqrRef(4,10:21)';
        xTrack = xy_ref_traj(:,1);
        yTrack = xy_ref_traj(:,2);
        
        % depending on time , recede horizon
    end

    % Time Varying Reference Update
    if k==1 && time(end)==0

        vRef = 0;
        [sxRef,syRef,phiRef] = genVehicleRef([X0(1:2)',[xTrack(1:N)';  yTrack(1:N)']], ...
                                             [xTrack(1:N+1)';         yTrack(1:N+1)']);
        phiRef = phi_ref_traj';
        Xref = [sxRef(1:N)', syRef(1:N)', phiRef(1:N)', vRef.*ones(size(sxRef(1:N)'))];
        
        input.x = [sxRef', syRef', phiRef', vRef.*ones(size(sxRef'))];
        
        input.y = [Xref(1:N,:), Uref];
        input.yN = Xref(N,:);
    end
    if time(end)>Ts % && mod(time(end),Ts*10)==0
        k = k + 1;

        vRef = 0;
        [sxRef,syRef,phiRef] = genVehicleRef([xTrack(k-1:k-1+N)'; yTrack(k-1:k-1+N)'], ...
                                             [    xTrack(k:k+N)';     yTrack(k:k+N)']);
        phiRef = phi_ref_traj';
        Xref = [sxRef(1:N)', syRef(1:N)', phiRef(1:N)', vRef.*ones(size(sxRef(1:N)'))];
        
        input.x = [sxRef', syRef', phiRef', vRef.*ones(size(sxRef'))];
        
        input.y = [Xref(1:N,:), Uref];
        input.yN = Xref(N,:);
    end

    % Solve NMPC OCP
    input.x0 = state_sim(end,:);
    output = autoraceMPCstep(input);

    % Save the MPC Step
    INFO_MPC = [INFO_MPC; output.info];
    KKT_MPC = [KKT_MPC; output.info.kktValue];
    controls_MPC = [controls_MPC; output.u(1,:)];
    input.x = output.x;
    input.u = output.u;
    
    % Simulate System
    sim_input.x = state_sim(end,:).';
    sim_input.u = output.u(1,:).';
    states = integrate_autorace(sim_input);
    state_sim = [state_sim; states.value'];
    input_sim = [input_sim; output.u(1,:)];
    
    iter = iter+1;
    nextTime = iter*Ts;
    time = [time, nextTime];
    disp(['Time: ',num2str(nextTime,'%06.1f'),char(9), ...
          'RTI step: ',num2str(output.info.cpuTime*1e3,'%.3f'),' ms ',char(9), ...
          'xRef=',num2str(Xref(1,1),'%07.4f'),char(9), ...
          'yRef=',num2str(Xref(1,2),'%07.4f'),char(9), ...
          'hRef=',num2str(rad2deg(Xref(end,3)),'%07.4f'),char(9), ...
          'err_d=',num2str(norm(Xref(1,[1,2])'-state_sim(1,[1,2])'),'%.4f'),char(9), ...
          'k=',num2str(k), ...
          'x =', num2str(sim_input.x(1)),...
          'y =', num2str(sim_input.x(2))]);
    
    if ROS_ENABLE == 1
        % Publish to high level
        msg =  rosmessage(pub);
        msg.Pose.Pose.Position.X = states.value(1);
        msg.Pose.Pose.Position.Y = states.value(2);
        msg.Twist.Twist.Linear.X = states.value(4);
        msg.Pose.Pose.Orientation.Z = states.value(3);
        send(pub, msg);
    end

%     visualize_learn;
%     pause(abs(0.1-toc));
    if k+N > length(xTrack), disp('End of Track'); break; end
end

%% Simulation - Plot
isSave = 0;
fWidth = 560;%480;1122
fHeight = 335;%240;420

tk = time';
XSim = state_sim';
XSim(3,:) = rad2deg(XSim(3,:));
XSim(4,:) = 3.6.*XSim(4,:);
USim = input_sim';
USim(1,:) = rad2deg(USim(1,:));

% States and Inputs Trajectories
% figure;
% for i=1:n
%     subplot(n,1,i);
%     plot(tk,XSim(i,:)');
%     if i==1, title(['$N($sec$) = ',num2str(N*Ts),'$'],'Interpreter','latex'); end
%     if i==n, xlabel('$t$','Interpreter','latex'); end
%     ylabel(sysStates(i),'Interpreter','latex');
%     grid on;
% end

% Trajectory Plot with Input Plots
% figure;
% subplot(2,2,[1,3]); hold on; grid on; axis equal; axis padded;
% plot(xTrack,yTrack,'-.');
% plot(XSim(1,:)',XSim(2,:)');
% plot(XSim(1,1), XSim(2,1),'ko');
% plot(Xref(:,1), Xref(:,2),'ro');
% title('Vehicle Trajectory','Interpreter','latex');
% xlabel(sysStates(1),'Interpreter','latex');
% ylabel(sysStates(2),'Interpreter','latex');
% for i=1:m
%     subplot(2,2,i*2);
%     plot(tk(1:end-1),USim(i,:)');
%     if i==1, title(['$N($sec$) = ',num2str(N*Ts),'$'],'Interpreter','latex'); end
%     if i==m, xlabel('$t$','Interpreter','latex'); end
%     ylabel(sysInputs(i),'Interpreter','latex');
%     grid on;
% end

% Trajectory Plot
wLine = 1;
figure; hold on; grid on; axis equal; axis padded;
title('Vehicle Trajectory','Interpreter','latex');
plot(xTrack,yTrack,'-.','LineWidth',wLine);
plot(XSim(1,:)',XSim(2,:)','LineWidth',wLine);
plot(XSim(1,1) ,XSim(2,1),'ko');

% Combination Plot
figure('Position',[230,240,1122,420]);
for i=1:n
    subplot(4,4,[i*4-3,i*4-3+1]);
    plot(tk,XSim(i,:)');
    if i==1, title(['$N($sec$) = ',num2str(N*Ts),'$'],'Interpreter','latex'); end
    if i==n, xlabel('$t$','Interpreter','latex'); end
    ylabel(sysStates(i),'Interpreter','latex');
    grid on;
end

subplot(4,4,[3,7,11,15]); hold on; grid on; axis equal; axis padded;
plot(xTrack,yTrack,'-.');
plot(XSim(1,:)',XSim(2,:)');
plot(XSim(1,1) ,XSim(2,1),'ko');
plot(Xref(:,1) ,Xref(:,2),'ro');
title('Vehicle Trajectory','Interpreter','latex');
xlabel(sysStates(1),'Interpreter','latex');
ylabel(sysStates(2),'Interpreter','latex');

for i=1:m
    subplot(4,4,8*(i-1)+[4,8]);
    plot(tk(1:end-1),USim(i,:)');
    if i==1, title(['$N($sec$) = ',num2str(N*Ts),'$'],'Interpreter','latex'); end
    if i==m, xlabel('$t$','Interpreter','latex'); end
    ylabel(sysInputs(i),'Interpreter','latex');
    grid on;
end

%% Helper functions
function [ X, Y ] = genTrackRef( R, n )

    theta = linspace(0,2*pi(),n)';
    X = R.*cos(theta);
    Y = R.*sin(theta);

end

function x = GetTraj(JointTrajectory)
    poses = reshape([JointTrajectory.points(:).positions],[3,21]);
%     poses = reshape(poses,[3,21])
%     x = poses(1,2:end);
%     y = poses(2,2:end);
%     phi = poses(3,2:end);
    v = [JointTrajectory.points(:).velocities];
%     v = v(2:end);
    x = [poses;v];
    % x is [x,y,phi,v] horizon sequence
end


