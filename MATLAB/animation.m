clear;

load("python/graph_out.mat", "saved_ff");
load("python/graph_out.mat", "input");

shape = size(saved_ff);
step = shape(1) / 20;
idx = 1;

fig = figure(1);

for i=[1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 300, 500, 700, 1000, 1300, 1500, 1700, 2000]
    angles = saved_ff(i, :);
    
    plot_v01(angles);
    title("Iteration = " + num2str(i));

    drawnow
    frame = getframe(fig);
    im{idx} = frame2im(frame);
    idx = idx + 1;
end
plot_v01(input);
title("Target");

drawnow
frame = getframe(fig);
im{idx} = frame2im(frame);

filename = "converge.gif"; % Specify the output file name
for idx_ = 1:idx
    [A,map] = rgb2ind(im{idx_},256);
    if idx_ == 1
        imwrite(A,map,filename,"gif","LoopCount",Inf,"DelayTime", 0.5);
    elseif idx_ == idx
        imwrite(A,map,filename,"gif","WriteMode","append","DelayTime", 1.5);
    else
        imwrite(A,map,filename,"gif","WriteMode","append","DelayTime", 0.5);
    end
end


function plot_v01(angles)
    % V1
    % links
    l1 = 155;
    l2 = 220;
    l3 = 45;
    l4 = 175;
    l5 = 115;
    
    % angles
    th1_ = angles(1);
    th2_ = angles(2);
    th3_ = angles(3);
    th4_ = angles(4);
    th5_ = angles(5);
    th6_ = angles(6);
    
    piOn2 = pi/2;
    
    T = {};
    T{1} = DH2tform(-th1_, l1, 0, piOn2);
    T{2} = DH2tform(th2_, 0, 0, -piOn2);
    T{3} = DH2tform(0, l2, 0, piOn2);
    T{4} = DH2tform(th3_, 0, 0, -piOn2);
    T{5} = DH2tform(-th4_, l3+l4, 0, piOn2);
    T{6} = DH2tform(piOn2+th5_, 0, 0, -piOn2);
    T{7} = DH2tform(th6_, l5, 0, 0);
    
    % Plotting
    points = [0 0 0];
    str_pts = ["0,0,0"];
    T_ = eye(4);
    plotTransforms([0 0 0], [1 0 0 0], "FrameSize", 50);
    axis([-800 800 -800 800 0 800])
    for i=1:length(T)
        T_ = T_ * T{i};
        r = T_(1:3, 1:3);
        t = transpose(T_(1:3, 4));
    
        r_q = rotm2quat(r);
    
        if (i == 5)
            disp(T_)
        end
        
        points = [points; t];
        str_pts = [str_pts strjoin(string(round(t,1)), ',')];
        
        hold on
        plotTransforms(t, r_q, "FrameSize", 50);
        axis([-200 200 -200 200 -200 200])
        hold off
    end
    
    hold on
    plot3(points(:,1), points(:,2), points(:,3), '-o', 'Color', 'black')
    plot3(points(end,1), points(end,2), points(end,3), '-o', 'Color', 'blue')
    text(points(:,1), points(:,2), points(:,3), str_pts)
    hold off
end