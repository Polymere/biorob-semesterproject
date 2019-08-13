clear all ;close all; clc
%% PATIENT 1 STRIKE TO STRIKE
load('C3D_angles.mat')
load('events_p1.mat')%414-> 637 

%%
% Slice left
plot(Angles.LAnkleAngles(:,1))
title("Left ankle")
sl_start=98;
sl_end=178;
hold on
plot(sl_start,Angles.LAnkleAngles(sl_start,1),'v')
plot(sl_end,Angles.LAnkleAngles(sl_end,1),'v')
legend(["Angle","Strike 1","Strike 2"]);
hold off
slice_left=sl_start:sl_end;
LANKLE=Angles.LAnkleAngles(slice_left,1);
LHIP=Angles.LHipAngles(slice_left,1);
LKNEE=Angles.LKneeAngles(slice_left,1);
figure()
plot(LANKLE)
hold on
plot(LHIP)
plot(LKNEE)
hold off
title("Left leg")
legend(["Ankle","Hip","Knee"])
%%
% Slice right

plot(Angles.RAnkleAngles(:,1))
title("Right ankle")
sl_start=138;
sl_end=218;
hold on
plot(sl_start,Angles.RAnkleAngles(sl_start,1),'v')
plot(sl_end,Angles.RAnkleAngles(sl_end,1),'v')
legend(["Angle","Strike 1","Strike 2"]);
hold off
slice_left=sl_start:sl_end;
RANKLE=Angles.RAnkleAngles(slice_left,1);
RHIP=Angles.RHipAngles(slice_left,1);
RKNEE=Angles.RKneeAngles(slice_left,1);
figure()
plot(RANKLE)
hold on
plot(RHIP)
plot(RKNEE)
hold off
title("Right leg")
legend(["Ankle","Hip","Knee"])

%
close all
figure()
plot(LANKLE)
hold on
plot(RANKLE)
legend(["Left","Right"])
title("Ankle")
hold off
figure()
plot(LHIP)
hold on
plot(RHIP)
legend(["Left","Right"])
title("Hip")
hold off
figure()
plot(LKNEE)
hold on
plot(RKNEE)
legend(["Left","Right"])
title("Knee")

%% PATIENT 2 STRIKE TO STRIKE
%p2
%215 -> 454
close all
clear all
load('Angles_ITW2.mat')
load('events_p2.mat')
%%
% Slice left
figure()
plot(Angles.LAnkleAngles(:,1))
title("Left ankle")
sl_start=121;
sl_end=200;
hold on
plot(sl_start,Angles.LAnkleAngles(sl_start,1),'v')
plot(sl_end,Angles.LAnkleAngles(sl_end,1),'v')
legend(["Angle","Strike 1","Strike 2"]);
hold off
slice_left=sl_start:sl_end;
LANKLE=Angles.LAnkleAngles(slice_left,1);
LHIP=Angles.LHipAngles(slice_left,1);
LKNEE=Angles.LKneeAngles(slice_left,1);

figure()
plot(LANKLE)
hold on
plot(LHIP)
plot(LKNEE)
hold off
title("Left leg")
legend(["Ankle","Hip","Knee"])
% Slice right
close all
plot(Angles.RAnkleAngles(:,1))
title("Right ankle")
sl_start=81;
sl_end=158;
hold on
plot(sl_start,Angles.RAnkleAngles(sl_start,1),'v')
plot(sl_end,Angles.RAnkleAngles(sl_end,1),'v')
legend(["Angle","Strike 1","Strike 2"]);
hold off
slice_left=sl_start:sl_end;
RANKLE=Angles.RAnkleAngles(slice_left,1);
RHIP=Angles.RHipAngles(slice_left,1);
RKNEE=Angles.RKneeAngles(slice_left,1);
figure()
plot(RANKLE)
hold on
plot(RHIP)
plot(RKNEE)
hold off
title("Right leg")
legend(["Ankle","Hip","Knee"])
close all
figure()
plot(LANKLE)
hold on
plot(RANKLE)
legend(["Left","Right"])
title("Ankle")
hold off
figure()
plot(LHIP)
hold on
plot(RHIP)
legend(["Left","Right"])
title("Hip")
hold off
figure()
plot(LKNEE)
hold on
plot(RKNEE)
legend(["Left","Right"])
title("Knee")

%% cmp winter

load("Winter_normal.mat")

plot(gait.data(:,2))


%% PATIENT 2 STRIKE TO LIFT

close all
clear all
load('Angles_ITW2.mat')
%% Slice left
figure()
plot(Angles.LAnkleAngles(:,1))
title("Left ankle")
sl_start=121;
sl_end=165;
hold on
plot(sl_start,Angles.LAnkleAngles(sl_start,1),'v')
plot(sl_end,Angles.LAnkleAngles(sl_end,1),'^')
legend(["Angle","Strike","Lift"]);
hold off
slice_left=sl_start:sl_end;
LANKLE=Angles.LAnkleAngles(slice_left,1);
LHIP=Angles.LHipAngles(slice_left,1);
LKNEE=Angles.LKneeAngles(slice_left,1);

figure()
plot(LANKLE)
hold on
plot(LHIP)
plot(LKNEE)
hold off
title("Left leg")
legend(["Ankle","Hip","Knee"])

%% Slice right
close all
plot(Angles.RAnkleAngles(:,1))
title("Right ankle")
sl_start=81;
sl_end=125;
hold on
plot(sl_start,Angles.RAnkleAngles(sl_start,1),'v')
plot(sl_end,Angles.RAnkleAngles(sl_end,1),'v')
legend(["Angle","Strike 1","Strike 2"]);
hold off
slice_left=sl_start:sl_end;
RANKLE=Angles.RAnkleAngles(slice_left,1);
RHIP=Angles.RHipAngles(slice_left,1);
RKNEE=Angles.RKneeAngles(slice_left,1);
figure()
plot(RANKLE)
hold on
plot(RHIP)
plot(RKNEE)
hold off
title("Right leg")
legend(["Ankle","Hip","Knee"])

close all
figure()
plot(LANKLE)
hold on
plot(RANKLE)
legend(["Left","Right"])
title("Ankle")
hold off
figure()
plot(LHIP)
hold on
plot(RHIP)
legend(["Left","Right"])
title("Hip")
hold off
figure()
plot(LKNEE)
hold on
plot(RKNEE)
legend(["Left","Right"])
title("Knee")

%% PATIENT 1 STRIKE TO LIFT
clear all;
load('C3D_angles.mat')
% Slice left
plot(Angles.LAnkleAngles(:,1))
title("Left ankle")
sl_start=98;
sl_end=144;
hold on
plot(sl_start,Angles.LAnkleAngles(sl_start,1),'v')
plot(sl_end,Angles.LAnkleAngles(sl_end,1),'^')
legend(["Angle","Strike","Lift"]);
hold off
slice_left=sl_start:sl_end;
LANKLE=Angles.LAnkleAngles(slice_left,1);
LHIP=Angles.LHipAngles(slice_left,1);
LKNEE=Angles.LKneeAngles(slice_left,1);
figure()
plot(LANKLE)
hold on
plot(LHIP)
plot(LKNEE)
hold off
title("Left leg")
legend(["Ankle","Hip","Knee"])
%%
% Slice right
close all
plot(Angles.RAnkleAngles(:,1))
title("Right ankle")
sl_start=138;
sl_end=184;
hold on
plot(sl_start,Angles.RAnkleAngles(sl_start,1),'v')
plot(sl_end,Angles.RAnkleAngles(sl_end,1),'^')
legend(["Angle","Strike","Lift"]);
hold off
slice_left=sl_start:sl_end;
RANKLE=Angles.RAnkleAngles(slice_left,1);
RHIP=Angles.RHipAngles(slice_left,1);
RKNEE=Angles.RKneeAngles(slice_left,1);
figure()
plot(RANKLE)
hold on
plot(RHIP)
plot(RKNEE)
hold off
title("Right leg")
legend(["Ankle","Hip","Knee"])

%%
close all
figure()
plot(LANKLE)
hold on
plot(RANKLE)
legend(["Left","Right"])
title("Ankle")
hold off
figure()
plot(LHIP)
hold on
plot(RHIP)
legend(["Left","Right"])
title("Hip")
hold off
figure()
plot(LKNEE)
hold on
plot(RKNEE)
legend(["Left","Right"])
title("Knee")


