clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Universal Automatic Elbow Detection for Lasso (non-uniform sampling)%%%
%%% Samuel Escudero, Luca Martino, Roberto San Millán,  Eduardo Morgado %%%           %%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%
%%% JUST RUN ME !!
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%
example=1;
disp('----------------------------------------------------')
%%%%%%%%%%%%%%%%%%%%
switch example
    case 1    
    load Curva_noth_err_out
    case 2
      load Curva_noth_err
      A=B;
     case 3
      load Curva_UNC_noth_err_out
      A=C;  
        case 4
      load Curva_UNC_noth_err
      A=D; 
        case 5
      load Curva_und_withth_err_2026
    %  A=D; 
end
disp('----------------------------------------------------')
V=A(end:-1:1,3);
kx=A(end:-1:1,2);
lambdaLASSO=A(end:-1:1,1);
%%%%%%%%%%%%%%%%%
V=V-min(V); %%% just a check, without loss of generality
%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda=V(1)/kx(end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% V(1)-V(1)/kmax*(k)=V(1)-lambda*k
aux=abs((V(1)-lambda*kx)-V);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pos_max=find(aux==max(max(aux)));
disp('----------------------------------------------------')
Elbow2=kx(pos_max);
disp(['RESULTS'])
disp('----------------------------------------------------')
disp(['Optimal number of links: ',num2str(Elbow2)])
disp('----------------------------------------------------')
%%%%%%%%%%%
figure
plot(kx,V,'o--','LineWidth',2,'MarkerSize',10)
hold on
plot(Elbow2,V(pos_max),'ro--','MarkerFaceColor','r','MarkerSize',10)
set(gca,'Fontsize',20)
%set(gca,'FontWeight','Bold')
xlabel('Number of links')
ylabel('\texttt{err}({\bf A}; {\bf X})','interpreter','latex')
box on
grid on

%%%%%%%%%%%%%%%
switch example
    case 1     
        text(27,0.68,'Looking just the','Fontsize',20)
         text(27,0.58,'connections of the two outputs','Fontsize',20)
         text(27,0.48,'with possible inter-link','Fontsize',20)
    case 2
         text(1027,0.68,'Considering all the connections','Fontsize',20)
         text(1027,0.58,'with all the variables','Fontsize',20)
         text(1027,0.48,'(with possible link within the','Fontsize',20)
         text(1027,0.38,'two outputs)','Fontsize',20)
    case 3
         text(28.2,0.68,'Looking just the','Fontsize',20)
         text(28.2,0.58,'connections of the two outputs','Fontsize',20)
         text(28.2,0.48,'WITHOUT possible inter-link','Fontsize',20)
    case 4 
          text(850,0.68,'Considering all the connections','Fontsize',20)
         text(850,0.58,'with all the variables','Fontsize',20)
         text(850,0.48,'(WITHOUT possible link within the','Fontsize',20)
         text(850,0.38,'two outputs)','Fontsize',20)
    case  5
     axis([100 2500 0 0.08])
      text(800,0.07,'Considering all the connections','Fontsize',20)
      text(800,0.06,'with all the variables','Fontsize',20)
      text(800,0.05,'(WITHOUT possible link within the','Fontsize',20)
      text(800,0.04,'two outputs)','Fontsize',20)
     
 end
 
 
%%%%annotation('textarrow',[0.19 0.15],[0.35 0.35],'FontSize',13,'Linewidth',2)
%%%%%annotation('textarrow',[0.35 0.15],[0.19 0.35],'FontSize',13,'Linewidth',2)
disp('----------------------------------------------------')

 disp('in lambda-of-LASSO values (optimal lambda):')
    disp(lambdaLASSO(pos_max))
 disp('----------------------------------------------------')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% uncertainty interval
INTest=Interval_AED_fun_LASSO(V,kx);
disp(' The  uncertainty interval (links):')
disp(INTest)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
posINTest1=find(kx==INTest(1));
posINTest2=find(kx==INTest(2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot([INTest(1) INTest(1)],[0 1],'r-','LineWidth',2)
plot([INTest(2) INTest(2)],[0 1],'r-','LineWidth',2)
 disp(' The interval in lambda-of-LASSO values:')
    disp([lambdaLASSO(posINTest1(1)) lambdaLASSO(posINTest2(1))])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
return
