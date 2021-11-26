%  MASWaves Inversion
%  Version: 06.2020
%%
%  e = MASWaves_misfit_MC(c_t,c_OBS)
%
%%
%  The function MASWaves_misfit_MC is used to evaluate the misfit between  
%  the theoretical and the experimental fundamental mode dispersion curves. 
%  The theoretical and experimental curves must be evaluated at the same 
%  wavelengths lambda. 
%
%% Input
%
%  Theoretical fundamental mode dispersion curve
%  c_t               Rayleigh wave phase velocity vector [m/s]
%
%  Experimental (observed) fundamental mode dispersion curve
%  c_OBS             Rayleigh wave phase velocity vector [m/s]
%
%% Output
%  e                 Dispersion misfit [%]
%
%% Subfunctions
%  (None)
%
%% References
%  MASWaves software
%  - Ólafsdóttir, E.Á. (2019). Multichannel Analysis of Surface Waves for
%    Soil Site Characterization. [Ph.D. dissertation]. Faculty of Civil
%    and Environmental Engineering, University of Iceland, 308 pp.
%    ISBN: 978-9935-9473-9-0.
%  - Ólafsdóttir, E.Á., Erlingsson, S. & Bessason, B. (2018). Tool for
%    analysis of multichannel analysis of surface waves (MASW) field data
%    and evaluation of shear wave velocity profiles of soils.
%    Canadian Geotechnical Journal, 55(2), 217–233. doi:10.1139/cgj-2016-0302.
%    [Open access]
%
%%
function  e = MASWaves_misfit_MC(c_t,c_OBS)

% Number of data points in theoretical and experimental dispersion curves.
Q = length(c_t); 

% Dispersion misfit
e = (1/Q)*sum(sqrt((c_OBS-c_t).^2)./c_OBS)*100;

end
