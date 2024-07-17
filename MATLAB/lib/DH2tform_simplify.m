function T = DH2tform_simplify(th, d, A, al)
    cos_th = trig_simplify('cos',th);
    cos_al = trig_simplify('cos',al);
    sin_th = trig_simplify('sin',th);
    sin_al = trig_simplify('sin',al);

    T = [
cos_th, -cos_al*sin_th,  sin_al*sin_th, A*cos_th;
sin_th,  cos_al*cos_th, -sin_al*cos_th, A*sin_th;
      0,          sin_al,          cos_al,         d;
      0,                0,                0,         1;
    ];
end