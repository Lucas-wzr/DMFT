#pragma once

#include <complex>

void polynomial_regression(
      std::complex<double>* fw, 
      double* freq, 
      const int nomega, 
      double& C1, 
      double& C2, 
      double& C3 );


//=================================
// Simpson integral got from ABACUS
// ================================
void Simpson_Integral(
      const int mesh,
      const double * const func,
      const double * const rab,
      double &asum );

void Simpson_Integral(
      const int mesh,
      const double * const func,
      const double dr,
      double &asum );

void Simpson_Integral_0toall(
      const int mesh,
      const double * const func,
      const double * const rab,
      double * const asum );

void Simpson_Integral_alltoinf(
      const int mesh,
      const double * const func,
      const double * const rab,
      double * const asum );
