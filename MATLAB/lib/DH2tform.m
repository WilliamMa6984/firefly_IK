function T = DH2tform(th, d, A, al)
    T = [
cos(th), -cos(al)*sin(th),  sin(al)*sin(th), A*cos(th);
sin(th),  cos(al)*cos(th), -sin(al)*cos(th), A*sin(th);
      0,          sin(al),          cos(al),         d;
      0,                0,                0,         1;
    ];
end