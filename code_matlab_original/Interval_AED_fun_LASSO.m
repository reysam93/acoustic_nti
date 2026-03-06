function INTest=Interval_AED_fun_LASSO(V,kx)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V=V-min(V); %%% just a check, without loss of generality
kmax=kx(end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
min_now=Inf;
x=kx;
x2=kx;
for i=1:length(x)
    for j=i+1:length(x)
         A1=((V(1)+V(i))*x(i))/2;
         A2=((V(i)+V(j))*(x2(j)-x(i)))/2;
         A3=(kmax-x2(j))*V(j)/2;
        E2=A1+A2+A3;
         if E2 < min_now 
             min_now=E2;
             INDICES_now=[kx(i),kx(j)];
         end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%
INTest=INDICES_now;
%%%%%%%%%%%%%%%%%%%%%%%%%





