%% Forward kinematics

clear;

l1 = 155;
l2 = 220;
l3 = 45;
l4 = 175-45;
l4_5 = 45;
l5 = 115;

%{
th1_ = -2.47475;
th2_ = -1.22313;
th3_ = 1.97023;
th4_ = -1.53234;
th5_ = -0.76750;
th6_ = -1.67948;
%}

%{
th1_ = rand(1)*pi*2 - pi;
th2_ = rand(1)*pi*2 - pi;
th3_ = rand(1)*pi*2 - pi;
th4_ = rand(1)*pi*2 - pi;
th5_ = rand(1)*pi*2 - pi;
th6_ = rand(1)*pi*2 - pi;
%}

angles = [pi/2 pi/2 pi/2 pi/2 pi/2 pi/2 pi/2 ];
th1_ = angles(1);
th2_ = angles(2);
th3_ = angles(3);
th4_ = angles(4);
th5_ = angles(5);
th5_5_ = angles(6);
th6_ = angles(7);


piOn2 = pi/2;

% V1

T = {};
T{1} = DH2tform(-th1_, l1, 0, piOn2);
T{2} = DH2tform(th2_, 0, 0, -piOn2);
T{3} = DH2tform(0, l2, 0, piOn2);
T{4} = DH2tform(th3_, 0, 0, -piOn2);
T{5} = DH2tform(-th4_, l3+l4, 0, piOn2);
T{6} = DH2tform(piOn2+th5_, 0, 0, 0);
T{7} = DH2tform(th5_5_, l4_5, 0, -piOn2); % T_6.5
T{8} = DH2tform(th6_, l5, 0, 0); % T_7

Tf_out = T{1}*T{2}*T{3}*T{4}*T{5}*T{6}*T{7}*T{8}

% Plotting

visibleJoints = [1 1 0 1 0 1 1 1];

points = [0 0 0];
str_pts = ["0,0,0"];
T_ = eye(4);
figure,
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
    
    
    %if (visibleJoints(i) == 1)
        points = [points; t];
        str_pts = [str_pts strjoin(string(round(t,1)), ',')];
        
        hold on
        plotTransforms(t, r_q, "FrameSize", 50);
        axis([-800 800 -800 800 0 800])
        hold off
    %end
    
end

hold on
plot3(points(:,1), points(:,2), points(:,3), '-o', 'Color', 'black')
plot3(points(end,1), points(end,2), points(end,3), '-o', 'Color', 'blue')
text(points(:,1), points(:,2), points(:,3), str_pts)
hold off

%% Symbolic

syms th1 th2 th3 th4 th5 th6

T = {};
T{1} = DH2tform_simplify(-th1, l1, 0, 'pi/2');
T{2} = DH2tform_simplify(th2, 0, 0, '-pi/2');
T{3} = DH2tform_simplify(0, l2, 0, 'pi/2');
T{4} = DH2tform_simplify(th3, 0, 0, '-pi/2');
T{5} = DH2tform_simplify(-th4, l3+l4, 0, 'pi/2');
T{6} = DH2tform_simplify('pi/2', 0, 0, 0); % for simplification
T{7} = DH2tform_simplify(th5, 0, 0, '-pi/2');
T{8} = DH2tform_simplify(th6, l5, 0, 0);

Tf_sym = T{1}*T{2}*T{3}*T{4}*T{5}*T{6}*T{7}*T{8}

%% Validate

th1 = th1_;
th2 = th2_;
th3 = th3_;
th4 = th4_;
th5 = th5_;
th6 = th6_;

%
Tf_sym_out = [sin(th6)*(cos(th4)*sin(th1) - sin(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) + cos(th6)*(sin(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - cos(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2))), cos(th6)*(cos(th4)*sin(th1) - sin(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - sin(th6)*(sin(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - cos(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2))), cos(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) + sin(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2)), 115*cos(th5)*(sin(th1)*sin(th4) + cos(th4)*(cos(th1)*sin(th2)*sin(th3) - cos(th1)*cos(th2)*cos(th3))) - 220*cos(th1)*sin(th2) + 115*sin(th5)*(cos(th1)*cos(th2)*sin(th3) + cos(th1)*cos(th3)*sin(th2)) - 220*cos(th1)*cos(th2)*sin(th3) - 220*cos(th1)*cos(th3)*sin(th2);
sin(th6)*(cos(th1)*cos(th4) + sin(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) + cos(th6)*(sin(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) + cos(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2))), cos(th6)*(cos(th1)*cos(th4) + sin(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) - sin(th6)*(sin(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) + cos(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2))), cos(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) - sin(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2)), 220*sin(th1)*sin(th2) + 115*cos(th5)*(cos(th1)*sin(th4) - cos(th4)*(sin(th1)*sin(th2)*sin(th3) - cos(th2)*cos(th3)*sin(th1))) - 115*sin(th5)*(cos(th2)*sin(th1)*sin(th3) + cos(th3)*sin(th1)*sin(th2)) + 220*cos(th2)*sin(th1)*sin(th3) + 220*cos(th3)*sin(th1)*sin(th2);
                                                                                                  cos(th6)*(cos(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - cos(th4)*sin(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2))) + sin(th4)*sin(th6)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)),                                                                                                   cos(th6)*sin(th4)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - cos(th4)*sin(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2))),                                                         - sin(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - cos(th4)*cos(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)),                                                                                220*cos(th2) + 220*cos(th2)*cos(th3) - 220*sin(th2)*sin(th3) - 115*sin(th5)*(cos(th2)*cos(th3) - sin(th2)*sin(th3)) - 115*cos(th4)*cos(th5)*(cos(th2)*sin(th3) + cos(th3)*sin(th2)) + 155;
                                                                                                                                                                                                                                                                                    0,                                                                                                                                                                                                                                                                                     0,                                                                                                                                                                      0,                                                                                                                                                                                                                                                                        1]
%

if (round(Tf_sym_out, -6) == round(Tf_out, -6))
    disp("Symbolic matrix is same as matrix multiply method")
end

