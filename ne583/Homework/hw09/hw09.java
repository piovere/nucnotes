class hw09
{
    static double[] wt;
    static double[] mu;
    static int nang;

    public static void main(String[] args)
    {
        try
        {
            double totxs=0.1;
            double alpha=0.8;
            double left=0.0;
            double right=0.0;

            int nx = 1000000;
            double width = 5. / totxs;
            double dx = width / nx;
            
            nang = 4;
            mu = new double[nang];
            wt = new double[nang];
            setQuadrature(nang);

            double[] scalar = new double[nx];

            for (int ix=0; ix < nx; ix++)
            {
                scalar[ix] = 0.0;
            }
            
            double wttot = 0.0;
            for (int ia=0; ia < nang / 2; ia++)
            {
                wttot += mu[ia] * wt[ia];
            }
            double influx = 1.0 / wttot;

            int inner = 0;
            double eps = 0.00001;
            double conv = 10000;
            double[] scalarOld = new double[nx];
            
            while ((Math.abs(conv) > eps) && (inner < 100000))
            {
                inner++;

                // copy scalar into scalarOld
                for (int ix=0; ix < nx; ix++)
                {
                    scalarOld[ix] = scalar[ix];
                    scalar[ix] = 0.0;
                }

                left = 0.;
                right = 0.;

                for (int ia=0; ia < nang; ia++)
                {
                    double phi0 = 0.0;
                    double muabs = Math.abs(mu[ia]);

                    if (mu[ia] < 0.0)
                    {
                        phi0 = 0.0;
                    }
                    else
                    {
                        phi0 = influx;
                    }

                    for (int ix0=0; ix0 < nx; ix0++)
                    {
                        int ix = 0;
                        if (mu[ia] < 0.0) {
                            ix = nx - 1 - ix0;
                        }

                        double phi1 = (muabs * phi0 / dx) / (muabs / dx + totxs);
                        double fluxave = phi1;
                        phi0 = phi1;

                        scalar[ix] += wt[ia] * fluxave;
                        
                    }
                    if (mu[ia] > 0.0) right += wt[ia] * phi0 * mu[ia];
                    if (mu[ia] < 0.0) left -= wt[ia] * phi0 * mu[ia];
                }

                // check inner convergence
                for (int ix=0; ix < nx; ix++)
                {
                    double etry=(scalar[ix] - scalarOld[ix]) / scalar[ix];
                    if (Math.abs(etry)<conv) conv = Math.abs(etry);
                }
            }

            System.out.println("Right leakage: " + right);
            System.out.println("Left leakage: " + left);
        }
        catch(Exception e)
        {
            e.printStackTrace(System.out);
        }
    }
    static void setQuadrature(int nang) throws Exception
    {
        if(nang==2)
        {
        wt[0]=1.;
        mu[0]=.5773502691;
        }
        else if(nang==4)
        {
        wt[0]=.6521451549;
        wt[1]=.3478548451;
        mu[0]=.3399810435;
        mu[1]=.8611363115;
        }
        else if(nang==8)
        {
        wt[0]=.3626837834;
        wt[1]=.3137066459;
        wt[2]=.2223810344;
        wt[3]=.1012285363;
        mu[0]=.1834346424;
        mu[1]=.5255324099;
        mu[2]=.7966664774;
        mu[3]=.9602898564;
        }
        else
        {
        throw new Exception(" Quadrature order must be 2,4 or 8, not "+nang);
        }
        double tot=0.;
        for(int ia=0;ia<nang/2;ia++)
        {
        mu[ia+nang/2]=-mu[ia];
        wt[ia]/=2.;
        wt[ia+nang/2]=wt[ia];
        tot+=wt[ia]+wt[ia+nang/2];
        }
        if(Math.abs(tot-1.)>.00001)throw new Exception("Wts add to "+tot);
    }
}