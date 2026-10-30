#include "KS_eigenvectors.h"
#include "../debug.h"
#include "../global_variables.h"

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>   //Use exit function
#include <iomanip>   //use setw and setprecison function

namespace DFT_output
{
  bool KS_eigenvectors::read_eigenvec(
          const std::string dir,
          const int ik,
          std::vector<DFT_plus_DMFT::Matrix>& eigenvector )
  {
    debug::codestamp("KS_eigenvectors::read_eigenvec");

    std::stringstream ss;

    int ispin, iband, ibasis;
    double real, imag;

    this->i_kpoint = ik;

    ss << dir << "eigenvector" << ik << ".dat";

    std::ifstream ifs(ss.str().c_str(), std::ios::in);

    if (!ifs)
	  {
	  	std::cerr << "Fail to oepn " << ss.str().c_str() << std::endl;
      std::exit(EXIT_FAILURE);
    }

    ifs.seekg(0);      //set the position at the beginning of the file
    ifs >> this->flag_SOC;
    ifs.ignore(150,'\n');

    if(flag_SOC) // SOC
    {

    }
    else //non_SOC
    {
      ifs >> this->nspins;
      ifs >> this->nbands;
      ifs >> this->nbasis;
      ifs.ignore(150, '\n');

      // One nbasis x selected-band matrix per spin.
      eigenvector.resize(this->nspins);
      for(ispin=0; ispin<nspins; ispin++)
        eigenvector[ispin].setZero(this->nbasis, this->nbands);

      while(ifs.good())
      {
        ifs >> ispin;
        if(ifs.eof()) break;
        ifs >> iband;
        ifs >> ibasis;

        ifs >> real;
        ifs >> imag;

        eigenvector[ispin](ibasis, iband) = std::complex<double>(real,imag);

        ifs.ignore(150, '\n'); 

        if(ifs.eof()) break;  //Check whether end of file is reached 
      }
    } 
    ifs.close();

    return true;
  }

  bool KS_eigenvectors::read_corr_subset(
      const std::string dir, const int ik,
      DFT_plus_DMFT::Hilbert_space& space,
      std::vector<DFT_plus_DMFT::Matrix>& eigenvector)
  {
    debug::codestamp("KS_eigenvectors::read_corr_subset");

    const std::vector<std::vector<int>>& correction=space.correction_flag();
    const std::vector<int>& wbands=space.Wbands();
    const std::vector<std::vector<int>>& index=space.ibands2wbands();

    std::stringstream ss;

    int ispin, iband, ibasis;
    double real, imag;

    this->i_kpoint = ik;

    ss << dir << "eigenvector" << ik << ".dat";

    std::ifstream ifs(ss.str().c_str(), std::ios::in);

    if (!ifs)
	  {
	  	std::cerr << "Fail to oepn " << ss.str().c_str() << std::endl;
      std::exit(EXIT_FAILURE);
    }

    ifs.seekg(0);      //set the position at the beginning of the file
    ifs >> this->flag_SOC;
    ifs.ignore(150,'\n');

    if(flag_SOC) // SOC
    {

    }
    else //non_SOC
    {
      ifs >> this->nspins;
      ifs >> this->nbands;
      ifs >> this->nbasis;
      ifs.ignore(150, '\n');

      // One nbasis x selected-band matrix per spin.
      eigenvector.resize(this->nspins);
      for(ispin=0; ispin<nspins; ispin++)
        eigenvector[ispin].setZero(this->nbasis, wbands[ispin]);

      while(ifs.good())
      {
        ifs >> ispin;
        if(ifs.eof()) break;
        ifs >> iband;
        ifs >> ibasis;

        ifs >> real;
        ifs >> imag;

        if(correction[ispin][iband])
          eigenvector[ispin](ibasis, index[ispin][iband]) = std::complex<double>(real,imag);

        ifs.ignore(150, '\n'); 

        if(ifs.eof()) break;  //Check whether end of file is reached 
      }
    } 
    ifs.close();

    return true;
  }

  bool KS_eigenvectors::read_DMFT_occ_subset(
      const std::string dir, const int ik,
      DFT_plus_DMFT::Hilbert_space& space,
      std::vector<DFT_plus_DMFT::Matrix>& eigenvector)
  {
    debug::codestamp("KS_eigenvectors::read_DMFT_occ_subset");

    const std::vector<std::vector<int>>& correction = space.correction_flag();
    const std::vector<int>& wbands=space.Wbands();
    const std::vector<std::vector<int>>& index = space.ibands2wbands();
    const std::vector<std::vector<int>>& wb2ib = space.wbands2ibands();

    std::stringstream ss;

    int ispin, iband, ibasis;
    double real, imag;

    this->i_kpoint = ik;

    ss << dir << "eigenvector" << ik << ".dat";

    std::ifstream ifs(ss.str().c_str(), std::ios::in);

    if (!ifs)
    {
      std::cerr << "Fail to oepn " << ss.str().c_str() << std::endl;
      std::exit(EXIT_FAILURE);
    }

    ifs.seekg(0);      //set the position at the beginning of the file
    ifs >> this->flag_SOC;
    ifs.ignore(150,'\n');

    if(flag_SOC) // SOC
    {

    }
    else //non_SOC
    {
      ifs >> this->nspins;
      ifs >> this->nbands;
      ifs >> this->nbasis;
      ifs.ignore(150, '\n');

      // One nbasis x selected-band matrix per spin.
      eigenvector.resize(this->nspins);
      for(ispin=0; ispin<nspins; ispin++)
        eigenvector[ispin].setZero(this->nbasis, (wb2ib[ispin].back()+1));

      while(ifs.good())
      {
        ifs >> ispin;
        if(ifs.eof()) break;
        ifs >> iband;
        ifs >> ibasis;

        ifs >> real;
        ifs >> imag;

        if( iband<(wb2ib[ispin].back()+1) )
          eigenvector[ispin](ibasis, iband) = std::complex<double>(real,imag);

        ifs.ignore(150, '\n'); 

        if(ifs.eof()) break;  //Check whether end of file is reached 
      }
    } 
    ifs.close();

    return true;
  }
  
  void KS_eigenvectors::evalute_k_wave_c_mat(
    DFT_plus_DMFT::Matrix& wave_c_mat, const int is,
    const int iband1, const int iband2,
    const std::vector<DFT_plus_DMFT::Matrix>& eigenvector )
  {
    const auto& C = eigenvector[is];
    wave_c_mat = C.col(iband1) * C.col(iband2).adjoint();
  }

}
