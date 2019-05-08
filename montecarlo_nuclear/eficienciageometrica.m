clear all;
%Script para calcular las eficiencias a partir del m�todo de Montecarlo. En
%todas ellas supondremos una fuente puntual.
%Datos experimentales
distancias=[15;20;25;30;35;40]; %en mm
Rd=5.64;
%Primera parte: c�lculo de las eficiencias geom�tricas
%mediante la expresi�n te�rica (para detector cil�ndrico).
eficiencia_teorica=zeros(1,length(distancias));
s_eficiencia_teorica=zeros(1,length(distancias));
for i=1:length(distancias)
    eficiencia_teorica(i)=1/2*(1-1/sqrt(1+(Rd/distancias(i))^2));
    s_eficiencia_teorica(i)=sqrt((Rd/(2*distancias(i)^2*(1+(Rd/distancias(i))^2)^1.5))^2*(ud^2*(Rd^2/distancias(i)^2)+sRd^2));
end
fprintf('Primera parte completada \n');
clear i

%Segunda parte: M�todo de Montecarlo con la fuente alineada con el eje del
%detector
n=1e8; %n es el n�mero de rayos que lanzo con el Montecarlo 
eficiencia_alineada=zeros(1,length(distancias));
for i=1:length(distancias)
    eficiencia_alineada(i)=montecarlo(distancias(i),Rd,n,0,0);
end
fprintf('Segunda parte completada \n');
clear i
% Tercera parte: a continuaci�n genero no s�lo una perturbaci�n en el eje Z (la
% %distancia), si no tambi�n en los ejes X e Y para comprobar el efecto que
% %tiene una alineaci�n indebida.
eficiencia_geometrica_XY=zeros(1,length(distancias));
for i=1:length(distancias) %i es el indice de la distancia
    x=normrnd(0,2);%Genero la perturbaci�n en el eje X.
    y=normrnd(0,2);%Genero la perturbaci�n en el eje Y.
    eficiencia_geometrica_XY(i)=montecarlo(distancias(i),Rd,n,x,y);
end
fprintf('Tercera parte completada \n');