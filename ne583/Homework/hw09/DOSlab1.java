import java.util.Scanner;
class ninedashtwo
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
      double sour=.4;
      int ng=1;
      int nx=10000;
      double dx=50./nx;
      Scanner sc=new Scanner(System.in);
      System.out.println(" No. of angles?");
      nang=sc.nextInt();
      mu=new double[nang];
      wt=new double[nang];
      setQuadrature();
      double[] scalar=new double[nx];
      for(int ix=0;ix<nx;ix++)
      {
        for(int ig=0;ig<ng;ig++)
        {
          scalar[ix]=0.;
        }
      }
      double[] sext=new double[nx];
      double[] sourin=new double[nx];
      for(int ig=0;ig<1;ig++)
      {
//**********************************************************************
//                                                                     *
//  Inner iterations                                                   *
//                                                                     *
//**********************************************************************
        int inner=0;
        double eps=0.00001;
        double conv=10000.;
        double[] scalarOld=new double[nx];
        while(Math.abs(conv)>eps)
        {
          inner++;

          // copy scalar into scalarOld
          for(int ix=0;ix<nx;ix++)
          {
            scalarOld[ix]=scalar[ix];
            scalar[ix]=0.;
          }
          left=0.;
          right=0.;
//**********************************************************************
//                                                                     *
//    Loop over directions                                             *
//                                                                     *
//**********************************************************************
          for(int ia=0;ia<nang;ia++)
          {
//**********************************************************************
//                                                                     *
//      Loop over positions                                            *
//                                                                     *
//**********************************************************************
            double phi0=0.;
            double muabs=Math.abs(mu[ia]);
            for(int ix0=0;ix0<nx;ix0++)
            {
              int ix=ix0;
              if(mu[ia]<0.)ix=nx-1-ix0;
//**********************************************************************
//                                                                     *
//        AUXILIARY: Find angular flux for cell and outgoing           *
//                                                                     *
//**********************************************************************
//                                                                     *
//          phi0 = Incoming angular flux                               *
//          phi1 = Outgoing angular flux                               *
//       fluxave = Average angular flux in cell                        *
//    sourin[ix] =  Source in the cell                                 *
//            mu = Absolute value of cosine of direction               *
//            dx = Width of the cell                                   *
//     totxs[ig] = Total cross section                                 *
//                                                                     *
//**********************************************************************
              if (ix==0)phi0=1.0;
              double phi1 = (muabs * phi0 / dx) / (muabs / dx + 0.1);
              double fluxave = phi1;
              phi0 = phi1;
//**********************************************************************
//                                                                     *
//        Add to scalar flux                                           *
//                                                                     *
//**********************************************************************
              scalar[ix]+=wt[ia]*fluxave;
            }
//**********************************************************************
//                                                                     *
//      Add to outgoing leakage                                        *
//                                                                     *
//**********************************************************************
            if(mu[ia]<0.)left-=wt[ia]*phi0*mu[ia];
            if(mu[ia]>0.)right+=wt[ia]*phi0*mu[ia];
          }
//**********************************************************************
//                                                                     *
//    Check inner convergence                                          *
//                                                                     *
//**********************************************************************
          conv=0.;
          for(int ix=0;ix<nx;ix++)
          {
            double etry=(scalar[ix]-scalarOld[ix])/scalar[ix];
            if(Math.abs(etry)>conv)conv=Math.abs(etry);
          }
        }
      }
//**********************************************************************
//                                                                     *
//    Print results                                                    *
//                                                                     *
//**********************************************************************
      
      for(int ig=0;ig<ng;ig++)
      {
        System.out.println("  Left grp "+(ig+1)+" is "+left);
      }
      for(int ig=0;ig<ng;ig++)
      {
        System.out.println(" Right grp "+(ig+1)+" is "+right);
      }
    }
    catch(Exception e)
    {
      e.printStackTrace(System.out);
    }
  }

  static void setQuadrature() throws Exception
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

