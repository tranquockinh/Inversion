%  MASWaves Inversion
%  Version: 06.2020
%%
%  F = MASWaves_FDMA(c,k,n,alpha,beta,rho,h)
%
%%
%  The function MASWaves_FDMA computes the value of the Rayleigh wave 
%  dispersion function F for the ordered couple (c,k). Computations are 
%  conducted using the fast delta matrix algorithm (FDMA) of Buchen and 
%  Ben-Hador (1996). 
%
%  The stratified soil model is described in terms of shear wave velocity,
%  compressional wave velocity, mass density and layer thicknesses. 
%
%% Input
%  c             Rayleigh wave phase velocity [m/s]
%  k             Wave number
%  n             Number of finite thickness layers
%  alpha         Compressional wave velocity vector [m/s] (array of length n+1)
%  beta          Shear wave velocity vector [m/s] (array of length n+1)
%  rho           Mass density vector [kg/m^3] (array of length n+1)
%  h             Layer thickness vector [m] (array of length n)
%
%% Output
%  F             F(c,k). Dispersion function value for the ordered couple (c,k).  
%
%% Subfunctions
%  MASWaves_FDMA_recursion
%
%% References
%  MASWaves software
%  - �lafsd�ttir, E.�. (2019). Multichannel Analysis of Surface Waves for 
%    Soil Site Characterization. [Ph.D. dissertation]. Faculty of Civil 
%    and Environmental Engineering, University of Iceland, 308 pp. 
%    ISBN: 978-9935-9473-9-0.
%  - �lafsd�ttir, E.�., Erlingsson, S. & Bessason, B. (2018). Tool for  
%    analysis of multichannel analysis of surface waves (MASW) field data 
%    and evaluation of shear wave velocity profiles of soils. 
%    Canadian Geotechnical Journal, 55(2), 217�233. doi:10.1139/cgj-2016-0302.
%    [Open access]
%  Fast delta matrix algorithm
%  - Buchen, P.W. & Ben-Hador, R. (1996). Free-mode surface-wave computations. 
%    Geophysical Journal International, 124(3), 869�887. 
%    doi:10.1111/j.1365-246X.1996.tb05642.x.
%
%%
function F = MASWaves_FDMA(c,k,n,alpha,beta,rho,h)

% Initialization
X = (rho(1)*beta(1)^2)^2 * [ 2*(2-c^2/beta(1)^2) , -(2-c^2/beta(1)^2)^2 , 0 , 0 , -4];

% Layer recursion (layers i = 1,...,n)
for i = 1:n
    X = MASWaves_FDMA_recursion(c,k,alpha(i),beta(i),beta(i+1),rho(i),rho(i+1),h(i),X);
end

% Dispersion function value
r = sqrt(1-c^2/alpha(n+1)^2);
s = sqrt(1-c^2/beta(n+1)^2);
F = X(2)+s*X(3)-r*(X(4)+s*X(5));

end


