clear all
close all
clc

N_Angles = 8;
[mu,wt] = Quadrature(N_Angles);

WtSum = 0;
for ii = 1:N_Angles/2
    WtSum = WtSum + mu(ii)*wt(ii);
end
Sour = 1/(WtSum);

totxs = [1];
NumCells = 10000;
dx = (5*(1/totxs))/NumCells;
%Sour = Sour/dx;
Alpha = 0.8;
NumGrp = 1;

Sext = zeros(NumCells,1);
%Sext(1) = Sour;
SourIn = Sext;
Scalar = zeros(NumCells,NumGrp);

Eps = 0.0001;
Error = 100000;

while Error > Eps
    
    ScalarOld = Scalar;
    Leakage = 0.0;
    
    for ia = 1:N_Angles
        
        Phi0 = Sour;
        Phi1 = 0.0;
        PhiAve = 0.0;
        
        for ix0 = 1:NumCells
            
            ix = ix0;
            SourceInFactor = 1.0;
            if mu(ia) < 0.0
                ix = NumCells + 1 - ix0;
                SourceInFactor = 0.0;
            end
            
            Phi1 = (SourIn(ix)*SourceInFactor + (abs(mu(ia))/dx - (1-Alpha)*totxs)*Phi0)/(abs(mu(ia))/dx + Alpha*totxs);
            PhiAve = (1-Alpha)*Phi0 + Alpha*Phi1;
            Phi0 = Phi1;
            Scalar(ix) = Scalar(ix) + wt(ia)*PhiAve;
            
        end
        
        if mu(ia) > 0.0
            
            Leakage = Leakage + wt(ia)*Phi0*mu(ia);
            
        end
        
    end
    
    Error = max(abs((Scalar-ScalarOld)./Scalar))
    
end