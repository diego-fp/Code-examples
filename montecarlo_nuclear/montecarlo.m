function [ eficiencia ] = montecarlo( d,R,C,x0,y0 )
%Simulaci�n Montecarlo (Diego Fern�ndez Prado)
%Variables de entrada:
%d=distancia entre la fuente y el detector
%R=radio del detector
%C=numero de cuentas lanzadas
%x0=posici�n en el eje X del centro del detector
%y0=posici�n en el eje Y del centro del detector
%Variables de salida
%eficienciaa=eficiencia simulada (total, geom�trica???****************)
N=0; %Inicio el n�mero de part�culas que entran en el detector
costheta=unifrnd(0,ones(1,C)); %Genero n�meros aleatorios entre -1 y 1 para el coseno de theta
phi=unifrnd(0,2*pi*ones(1,C)); %Genero n�meros aleatorios entre 0 y 2*pi para phi
x=x0-d.*cos(phi).*sqrt(1-costheta.^2)./costheta; %Calculo el valor de la coordenada x
y=y0-d.*sin(phi).*sqrt(1-costheta.^2)./costheta; %Calculo el valor de la coordenada y
for i=1:C
    if x(i)^2+y(i)^2<R^2
        N=N+1;
    end
end
 
eficiencia=N/(2*C);

end