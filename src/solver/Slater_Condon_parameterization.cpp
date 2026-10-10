#include "Slater_Condon_parameterization.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <sstream>
#include <stdexcept>

namespace
{
  double factorial(int n)
  {
    double value=1.0;
    for(int i=2;i<=n;i++) value*=i;
    return value;
  }
  double phase(int n) { return n%2==0 ? 1.0 : -1.0; }

  // Racah finite sum, integer angular momenta only (l <= 3).
  double three_j(int a,int b,int c,int x,int y,int z)
  {
    if(x+y+z!=0 || std::abs(x)>a || std::abs(y)>b || std::abs(z)>c
       || c> a+b || c<std::abs(a-b)) return 0.0;
    double norm=phase(a-b-z)*std::sqrt(factorial(a+b-c)*factorial(a-b+c)
      *factorial(-a+b+c)/factorial(a+b+c+1)
      *factorial(a+x)*factorial(a-x)*factorial(b+y)*factorial(b-y)
      *factorial(c+z)*factorial(c-z));
    double sum=0.0;
    int first=std::max(0,std::max(b-c-x,a-c+y));
    int last=std::min(a+b-c,std::min(a-x,b+y));
    for(int t=first;t<=last;t++)
      sum+=phase(t)/(factorial(t)*factorial(a+b-c-t)*factorial(a-x-t)
        *factorial(b+y-t)*factorial(c-b+x+t)*factorial(c-a-y+t));
    return norm*sum;
  }
}

namespace DMFT
{
  namespace slater_condon
  {
    std::vector<double> from_U_J(int l,double U,double J,double f4_f2,double f6_f2)
    {
      if((l<1 || l>3) || !std::isfinite(U) || !std::isfinite(J) || U<0 || J<0)
        throw std::runtime_error("Invalid shell or shell-averaged U,J");
      // TRIQS U_J_to_radial_integrals defaults; ratios are modeling assumptions.
      if(f4_f2==-1.0)
      {
        if(l==1) f4_f2=0.0;
        else if(l==2) f4_f2=0.63;
        else f4_f2=451.0/675.0;
      }
      if(f6_f2==-1.0)
      {
        if(l==3) f6_f2=1001.0/2025.0;
        else f6_f2=0.0;
      }
      if(!std::isfinite(f4_f2) || !std::isfinite(f6_f2) || f4_f2<0 || f6_f2<0
        || (l==1 && f4_f2!=0.0) || (l<=2 && f6_f2!=0.0))
        throw std::runtime_error("Invalid Slater integral ratios");
      std::vector<double> F(l+1,0.0);
      F[0]=U;
      if(l==1) F[1]=5.0*J;
      else if(l==2) F[1]=14.0*J/(1.0+f4_f2);
      else F[1]=6435.0*J/(286.0+195.0*f4_f2+250.0*f6_f2);
      if(l>=2) F[2]=f4_f2*F[1];
      if(l==3) F[3]=f6_f2*F[1];
      return F;
    }

    // Slater_integrals_impurityN.dat (N: zero-based inequivalent impurity):
    // one entry per line, e.g. "F0 8.0", in eV; no count header; # starts a comment.
    // Required: p={F0,F2}, d={F0,F2,F4}, f={F0,F2,F4,F6}, in any order.
    // Values must be finite and nonnegative; missing/duplicate entries are errors.
    std::vector<double> read_integrals(std::istream& input,int l,double energy_unit)
    {
      if(l<1 || l>3) throw std::runtime_error("Slater-Condon requires a p, d or f shell");
      std::vector<double> F(l+1,0.0);
      std::vector<bool> found(l+1,false);
      std::string line;
      while(std::getline(input,line))
      {
        line=line.substr(0,line.find('#'));
        std::istringstream row(line);
        std::string key,extra;
        if(!(row>>key)) continue;
        double value;
        if(key.size()!=2 || key[0]!='F' || key[1]<'0' || key[1]>'6'
          || (key[1]-'0')%2!=0 || (key[1]-'0')/2>l
          || !(row>>value) || (row>>extra) || !std::isfinite(value) || value<0.0)
          throw std::runtime_error("Invalid Slater integral line: "+line);
        int k=(key[1]-'0')/2;
        if(found[k]) throw std::runtime_error("Duplicate Slater integral "+key);
        found[k]=true;
        F[k]=value/energy_unit;
      }
      for(bool present:found)
        if(!present) throw std::runtime_error("Specify every F0,F2,...,F2l explicitly");
      return F;
    }

    rutgers_atomic::tensor evaluate(int l,const std::vector<double>& F,
                                   const std::vector<int>& orbitals)
    {
      if(orbitals.empty())
        throw std::runtime_error("Empty Slater-Condon orbital list");
      std::vector<bool> used(2*l+1,false);
      for(int m:orbitals)
      {
        if(std::abs(m)>l || used[m+l]) throw std::runtime_error("Invalid real harmonic labels");
        used[m+l]=true;
      }
      const int M=orbitals.size(), L=2*M, S=2*l+1;
      typedef std::complex<double> complex;
      // |R_m> = sum_n T[m][n] |Y_ln>.
      std::vector<std::vector<complex>> T(M,std::vector<complex>(S,0.0));
      for(int a=0;a<M;a++)
      {
        int m=orbitals[a], p=std::abs(m);
        if(m==0) T[a][l]=1.0;
        else if(m>0)
        {
          T[a][l+p]=phase(p)/std::sqrt(2.0);
          T[a][l-p]=1.0/std::sqrt(2.0);
        }
        else
        {
          T[a][l+p]=complex(0,-phase(p)/std::sqrt(2.0));
          T[a][l-p]=complex(0,1.0/std::sqrt(2.0));
        }
      }
      std::vector<std::vector<std::vector<complex>>> C(2*l+1);
      // <R_a|C^k_q|R_c>, C^k_q=sqrt(4pi/(2k+1))Y_kq.
      for(int k=0;k<=2*l;k+=2)
      {
        C[k].assign(2*k+1,std::vector<complex>(M*M,0.0));
        for(int q=-k;q<=k;q++) for(int a=0;a<M;a++) for(int c=0;c<M;c++)
          for(int m=-l;m<=l;m++) for(int n=-l;n<=l;n++)
            C[k][q+k][a*M+c]+=std::conj(T[a][m+l])*T[c][n+l]
              *phase(m)*double(2*l+1)*three_j(l,k,l,0,0,0)*three_j(l,k,l,-m,q,n);
      }
      auto U=rutgers_atomic::tensor(L,std::vector<std::vector<std::vector<double>>>(L,
        std::vector<std::vector<double>>(L,std::vector<double>(L,0.0))));
      for(int a=0;a<M;a++) for(int b=0;b<M;b++)
        for(int c=0;c<M;c++) for(int d=0;d<M;d++)
        {
          complex value=0.0;
          for(int k=0;k<=2*l;k+=2) for(int q=-k;q<=k;q++)
            value+=F[k/2]*phase(q)*C[k][q+k][a*M+c]*C[k][-q+k][b*M+d];
          if(std::abs(value.imag())>1e-12)
            throw std::runtime_error("Complex tensor in real harmonic basis");
          // Spatial V_ab;cd maps to U[a sigma,b tau,d tau,c sigma].
          // Do not antisymmetrize: fermionic operators supply exchange.
          double v=std::abs(value.real())<1e-14 ? 0.0 : value.real();
          for(int s=0;s<2;s++) for(int t=0;t<2;t++)
            U[a+s*M][b+t*M][d+t*M][c+s*M]=v;
        }
      return U;
    }
  }
}
