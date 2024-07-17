function ans = trig_simplify(func, th)
    if (strcmp(th, 'pi/2'))
        if (strcmp(func, 'cos'))
            ans = 0;
        elseif (strcmp(func, 'sin'))
            ans = 1;
        end
    elseif (strcmp(th, '-pi/2'))
        if (strcmp(func, 'cos'))
            ans = 0;
        elseif (strcmp(func, 'sin'))
            ans = -1;
        end
    else
        % Not simplified
        if (strcmp(func, 'cos'))
            ans = cos(th);
        elseif (strcmp(func, 'sin'))
            ans = sin(th);
        end
    end
end