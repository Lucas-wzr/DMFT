#include "projector.h"
#include "../mpi_environment.h"
#include "../para/overlap_matrix.h"
#include "../matrix_types.h"
#include <Eigen/Dense>
#include "../constants.h"
#include "../global_variables.h"
#include "../timer.h"
#include "../debug.h"

#include <mpi.h>


//test
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace DFT_plus_DMFT
{
  void projector::evaluate_projector(
        const int projector_type,
        const int DFT_solver,
        DFT_output::KS_bands& band,
        DFT_plus_DMFT::Hilbert_space& space,
        DFT_output::atoms_info& atom,
        const double CWF_ep0,
        const double CWF_ep1,
        const double CWF_T,
        const double projection_min_weight )
  {
    debug::codestamp("projector::evaluate_projector");

    double time;
    double seconds;
    timer::timestamp(time);

    const std::vector<int>& wbands = space.Wbands();
    const std::vector<int>& norb_sub = atom.iatom_norb();
    const std::vector<std::vector<int>>& orb_index = atom.Im2iorb();
    const int natom = atom.total_atoms();
    const int nkpoints = band.nk();
    const int nspin = band.nspins();
    const int nbands = band.nband();
    const double Efermi = band.Fermi_level();
    const auto& epsilon = space.eigen_val();
    const int norb = atom.norb();
    const std::complex<double> zero(0.0,0.0), one(1.0,0.0);
    const int myid = mpi_rank();
    const int ntasks = mpi_ntasks();

    const auto projector_abort =
        [myid](const std::string& message,
               const int global_k,
               const int ispin,
               const int info)
    {
      std::cerr << "[projector fatal]"
                << " rank=" << myid
                << " global_k=" << global_k
                << " spin=" << ispin
                << " solver_info=" << info
                << " : " << message << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      std::abort();
    };

    const auto hermitian_sqrt_inverse =
        [&](const Eigen::Ref<const Matrix>& matrix,
            Matrix& inverse_sqrt,
            const int global_k,
            const int ispin,
            const std::string& label)
    {
      Eigen::SelfAdjointEigenSolver<Matrix> solver;
      solver.compute(matrix.selfadjointView<Eigen::Upper>());
      if(solver.info() != Eigen::Success)
        projector_abort(label + " eigensolver failed", global_k, ispin, solver.info());

      const auto& eigenvalues = solver.eigenvalues();
      for(int i=0; i<eigenvalues.size(); i++)
        if(!std::isfinite(eigenvalues[i]) || !(eigenvalues[i] > 0.0) ||
           std::fabs(eigenvalues[i]) < 1.0e-12)
          projector_abort(label + " is not positive definite",
                          global_k, ispin, 0);

      inverse_sqrt = solver.eigenvectors()
          * eigenvalues.array().sqrt().inverse().matrix().asDiagonal()
          * solver.eigenvectors().adjoint();
    };

    if(projector_type != 1 && projector_type != 2)
      projector_abort("unsupported projector type", -1, -1, 0);

    if(ntasks <= 0 || nspin <= 0 || norb <= 0 || natom <= 0 ||
       nkpoints <= 0 || nbands <= 0 ||
       wbands.size() != static_cast<std::size_t>(nspin))
      projector_abort("invalid global projector dimensions", -1, -1, 0);

    if(norb_sub.size() != static_cast<std::size_t>(natom) ||
       orb_index.size() != static_cast<std::size_t>(natom) ||
       epsilon.size() != static_cast<std::size_t>(nspin))
      projector_abort("invalid projector container dimensions", -1, -1, 0);

    int norb_count = 0;
    for(int iatom=0; iatom<natom; iatom++)
    {
      const int m_tot = norb_sub[iatom];
      if(m_tot <= 0 ||
         orb_index[iatom].size() != static_cast<std::size_t>(m_tot))
        projector_abort("invalid correlated-atom dimensions", -1, -1, 0);

      norb_count += m_tot;
      for(int m=0; m<m_tot; m++)
        if(orb_index[iatom][m] < 0 || orb_index[iatom][m] >= norb)
          projector_abort("correlated-orbital index is out of range", -1, -1, 0);
    }

    if(norb_count != norb)
      projector_abort("inconsistent correlated-orbital count", -1, -1, 0);

    for(int is=0; is<nspin; is++)
    {
      if(wbands[is] <= 0 || wbands[is] > nbands)
        projector_abort("invalid correlated KS-band count", -1, is, 0);
      if(projector_type == 2 && wbands[is] < norb)
        projector_abort("CWF requires wbands >= norb", -1, is, 0);
      if(epsilon[is].size() != static_cast<std::size_t>(nkpoints))
        projector_abort("invalid KS-energy k-point dimension", -1, is, 0);
      for(int ik=0; ik<nkpoints; ik++)
        if(epsilon[is][ik].size() != static_cast<std::size_t>(wbands[is]))
          projector_abort("invalid KS-energy band dimension", ik, is, 0);
    }

    if(projector_type == 2 &&
       (!std::isfinite(CWF_ep0) || !std::isfinite(CWF_ep1) ||
        !std::isfinite(CWF_T) || !(CWF_ep0 < CWF_ep1) || !(CWF_T > 0.0)))
      projector_abort("invalid CWF parameters", -1, -1, 0);

    const int wbands_max = *std::max_element(wbands.begin(), wbands.end());

    std::vector<int> k_map;
    for(int ik=0; ik<nkpoints; ik++)
      if(ik%ntasks == myid) k_map.push_back(ik);

    // Rebuild the exact local projector shapes on every call so no values
    // from an earlier evaluation can survive in persistent storage.
    this->projector_mat.resize(k_map.size());
    this->TB_Hk.resize(k_map.size());
    this->unitary_trans.resize(k_map.size());
    for(std::size_t ik=0; ik<k_map.size(); ik++)
    {
      this->projector_mat[ik].resize(natom);
      this->TB_Hk[ik].resize(natom);
      this->unitary_trans[ik].resize(natom);
      for(int iatom=0; iatom<natom; iatom++)
      {
        const int m_tot = norb_sub[iatom];
        this->projector_mat[ik][iatom].resize(nspin);
        this->TB_Hk[ik][iatom].resize(nspin);
        this->unitary_trans[ik][iatom].resize(nspin);
        for(int is=0; is<nspin; is++)
        {
          this->projector_mat[ik][iatom][is] = Matrix::Zero(wbands[is], m_tot);
          this->TB_Hk[ik][iatom][is].assign(m_tot, zero);
          this->unitary_trans[ik][iatom][is] = Matrix::Zero(m_tot, m_tot);
        }
      }
    }

    DFT_output::overlap_matrix ovlp(DFT_solver, "dft/outputs_to_DMFT/overlap_matrix");
    wave_function wfc;

    Matrix Lowdin(norb, norb);
    Matrix proj_mat_tmp, mat_tmp;
    Matrix norm_orth(norb, norb);
    Matrix sqrt_inver(norb, norb);
    std::vector<double> window_weight(projector_type == 2 ? wbands_max : 0);

    double cwf_energy_min_local = std::numeric_limits<double>::infinity();
    double cwf_energy_max_local = -std::numeric_limits<double>::infinity();
    double cwf_window_min_local = std::numeric_limits<double>::infinity();
    double cwf_window_max_local = 0.0;
    double cwf_rcond_min_local = std::numeric_limits<double>::infinity();
    double cwf_orth_max_local = 0.0;
    double cwf_hermitian_max_local = 0.0;

    std::vector<Matrix> eigenvec;
    std::ofstream window_report("window_overlap_rank" + std::to_string(myid) + ".dat");
    if(!window_report)
      projector_abort("cannot open window overlap diagnostics", -1, -1, 0);
    window_report << "# k spin threshold eigenvalues(A^dagger A), ascending; before normalization and CWF weights\n"
                  << std::setprecision(17);
    struct { double weight; int point; } local_min = {std::numeric_limits<double>::infinity(), -1}, global_min;
    int rejected_local = 0;

    const auto check_window_overlap = [&](int i_k_point, int is)
    {
      // A contains overlaps of the KS window with Lowdin-orthonormal local orbitals.
      // Check coverage before normalization can hide weak or missing directions.
      norm_orth = mat_tmp.adjoint() * mat_tmp;
      Eigen::SelfAdjointEigenSolver<Matrix> solver;
      solver.compute(norm_orth.selfadjointView<Eigen::Upper>(), Eigen::EigenvaluesOnly);
      if(solver.info() != Eigen::Success)
        projector_abort("window overlap eigensolver failed", i_k_point, is, solver.info());
      const auto& eigen_val = solver.eigenvalues();
      window_report << i_k_point << " " << is << " " << projection_min_weight;
      for(double value : eigen_val){
        if(!std::isfinite(value))
          projector_abort("non-finite window overlap eigenvalue", i_k_point, is, 0);
        window_report << " " << value;
      }
      window_report << "\n";
      if(eigen_val[0] < local_min.weight){
        local_min.weight = eigen_val[0];
        local_min.point = i_k_point*nspin + is;
      }
      if(eigen_val[0] < projection_min_weight){
        rejected_local++;
        return false;
      }
      return true;
    };

    const auto finish_window_overlap = [&]()
    {
      window_report.close();
      int rejected = 0;
      MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
      MPI_Allreduce(&rejected_local, &rejected, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if(myid == 0){
        GLV::ofs_running << "Window overlap: minimum weight = " << std::setprecision(12)
                         << global_min.weight << ", k = " << global_min.point/nspin
                         << ", spin = " << global_min.point%nspin
                         << ", threshold = " << projection_min_weight
                         << ", rejected k/spin points = " << rejected << std::endl;
      }
      if(rejected){
        std::ostringstream message;
        message << "KS window does not cover the correlated subspace: minimum weight "
                << global_min.weight << " < " << projection_min_weight
                << "; enlarge projection_window";
        projector_abort(message.str(), global_min.point/nspin, global_min.point%nspin, 0);
      }
    };

    if(myid == 0)
      GLV::ofs_running << (projector_type == 2
          ? "\n=============Projector(CWF)============"
          : "\n=============Projector(default)============") << std::endl;

    for(int ik=0; ik<k_map.size(); ik++)
    {
      const int i_k_point = k_map[ik];

      ovlp.evaluate_ovlp_k(i_k_point, atom);        //caculate overlap matrix in k-space

      wfc.read_corr_subset("dft/outputs_to_DMFT/KS_eigenvector/", i_k_point, space, eigenvec);       //read KS-eigenvector

      this->n_basis=wfc.basis_n();
      
      const Matrix& overlap_columns = ovlp.overlap();

      const Matrix& local_overlap = ovlp.overlap_localorb();

      if(this->n_basis <= 0 ||
         eigenvec.size() != static_cast<std::size_t>(nspin) ||
         overlap_columns.rows() != this->n_basis || overlap_columns.cols() != norb ||
         local_overlap.rows() != norb || local_overlap.cols() != norb)
        projector_abort("invalid projector matrix dimensions", i_k_point, -1, 0);

      hermitian_sqrt_inverse(
          local_overlap, Lowdin,
          i_k_point, -1, "local overlap matrix");

      for(int is=0; is<nspin; is++)
      {
        if(eigenvec[is].rows() != this->n_basis || eigenvec[is].cols() != wbands[is])
          projector_abort("invalid KS-eigenvector dimensions", i_k_point, is, 0);

        // void cblas_zgemm (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const
        // CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n, const MKL_INT k, const void
        // *alpha, const void *a, const MKL_INT lda, const void *b, const MKL_INT ldb, const void
        // *beta, void *c, const MKL_INT ldc);

        //===============test the orthonormality of wavefunctions=====
        /*
        std::vector<std::complex<double>> wfc_norm(wbands[is]*wbands[is]);
        std::vector<std::complex<double>> wfc_ovlp_tmp(wbands[is]*this->n_basis);
        const std::vector<std::complex<double>>& ovlp_all = ovlp.ovlp_aims.ovlp_mat_work();
        cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
                    wbands[is], this->n_basis, this->n_basis,
                    &one,
                    &eigenvec[is][0], wbands[is],
                    &ovlp_all[0], this->n_basis,
                    &zero,
                    &wfc_ovlp_tmp[0], this->n_basis );

        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    wbands[is], wbands[is], this->n_basis,
                    &one,
                    &wfc_ovlp_tmp[0], this->n_basis,
                    &eigenvec[is][0], wbands[is],
                    &zero,
                    &wfc_norm[0], wbands[is] );
        
        std::stringstream ss;
        ss << "wfc-norm/wfc_norm_is" << is << "_ik" << ik << ".dat";
        std::ofstream ofs(ss.str().c_str(), std::ios::out);

        for(int m1=0; m1<wbands[is]; m1++)
        {
          for(int m2=0; m2<wbands[is]; m2++)
          {
            ofs << std::setw(15) << std::fixed << std::setprecision(9) << wfc_norm[m1*wbands[is]+m2].real()
                << std::setw(15) << std::fixed << std::setprecision(9) << wfc_norm[m1*wbands[is]+m2].imag(); 
          }
          ofs << std::endl;
        }//m
        ofs.close(); 
        */

        const Matrix& C = eigenvec[is];
        proj_mat_tmp = C.adjoint() * overlap_columns;
        mat_tmp = proj_mat_tmp * Lowdin;

        if(!check_window_overlap(i_k_point,is)) continue;

        if(projector_type == 1)
        {
          norm_orth = mat_tmp.adjoint() * mat_tmp;
          hermitian_sqrt_inverse(
              norm_orth, sqrt_inver,
              i_k_point, is, "default projector Gram matrix");
        /*
        for(int i=0; i<norb; i++){
          if(std::fabs(eigen_val[i])<1.0e-12){
            std::cout << i_k_point << "  "
                      << std::setprecision(3) 
                      << std::setiosflags(std::ios::scientific) 
                      << eigen_val[i] << std::endl;
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    wbands[is], norb, norb,
                    &one,
                    &sqrt_inver[0], norb,
                    &sqrt_inver[0], norb,
                    &zero,
                    &test_mat[0], norb);
            
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    wbands[is], norb, norb,
                    &one,
                    &test_mat[0], norb,
                    &norm_orth[0], norb,
                    &zero,
                    &sqrt_inver[0], norb);

            for(int iorb1=0; iorb1<norb; iorb1++){
              for(int iorb2=0; iorb2<norb; iorb2++){
                std::cout << std::setw(7) << std::fixed << std::setprecision(3) << sqrt_inver[iorb1*norb+iorb2].real()
                          << std::setw(7) << std::fixed << std::setprecision(3) << sqrt_inver[iorb1*norb+iorb2].imag(); 
              }
              std::cout << std::endl;
            }
          }
        }
        */

          proj_mat_tmp = mat_tmp * sqrt_inver;
        }
        else
        {
          // DF stores absolute KS energies in Hartree. DMFT.in CWF bounds are
          // in eV relative to E_F, so convert only the window boundaries; the
          // local Hamiltonian below deliberately remains on the absolute scale.
          const double epsi0 = Efermi + CWF_ep0/GLC::Hartree_to_eV;
          const double epsi1 = Efermi + CWF_ep1/GLC::Hartree_to_eV;
          const double beta = GLC::Hartree_to_eV/CWF_T;

          if(!std::isfinite(epsi0) || !std::isfinite(epsi1) ||
             !std::isfinite(beta) || !(epsi0 < epsi1))
            projector_abort("non-finite CWF energy conversion", i_k_point, is, 0);

          double max_window = 0.0;
          for(int iband=0; iband<wbands[is]; iband++)
          {
            const double energy = epsilon[is][i_k_point][iband];
            if(!std::isfinite(energy))
              projector_abort("non-finite KS energy", i_k_point, is, 0);

            const double relative_energy_eV =
                (energy - Efermi)*GLC::Hartree_to_eV;
            cwf_energy_min_local =
                std::min(cwf_energy_min_local, relative_energy_eV);
            cwf_energy_max_local =
                std::max(cwf_energy_max_local, relative_energy_eV);

            const double lower_exponent = beta*(epsi0 - energy);
            const double upper_exponent = beta*(energy - epsi1);
            double weight = 0.0;
            if(lower_exponent < 25.0 && upper_exponent < 25.0)
              weight = 1.0/(1.0 + std::exp(lower_exponent))
                     + 1.0/(1.0 + std::exp(upper_exponent)) - 1.0;

            const double weight_tolerance =
                64.0*std::numeric_limits<double>::epsilon();
            if(!std::isfinite(weight) || weight < -weight_tolerance ||
               weight > 1.0 + weight_tolerance)
              projector_abort("invalid CWF window weight", i_k_point, is, 0);

            window_weight[iband] = std::max(0.0, std::min(1.0, weight));
            max_window = std::max(max_window, window_weight[iband]);
            cwf_window_min_local =
                std::min(cwf_window_min_local, window_weight[iband]);
            cwf_window_max_local =
                std::max(cwf_window_max_local, window_weight[iband]);

            mat_tmp.row(iband) *= window_weight[iband];
          }

          if(!(max_window > 0.0))
            projector_abort("CWF window contains no usable KS states", i_k_point, is, 0);

          Eigen::JacobiSVD<Matrix> svd(mat_tmp, Eigen::ComputeThinU | Eigen::ComputeThinV);
          if(svd.info() != Eigen::Success)
            projector_abort("CWF SVD did not converge", i_k_point, is, svd.info());
          const auto& singular_values = svd.singularValues();

          for(int i=0; i<norb; i++)
            if(!std::isfinite(singular_values[i]))
              projector_abort("non-finite CWF singular value", i_k_point, is, 0);

          const double s_max = singular_values[0];
          const double s_min = singular_values[static_cast<std::size_t>(norb - 1)];
          const double rank_tolerance =
              std::numeric_limits<double>::epsilon()*
              static_cast<double>(std::max(wbands[is], norb))*s_max;

          if(!(s_max > 0.0) || s_min <= rank_tolerance)
          {
            std::cerr << "[CWF singular spectrum]"
                      << " rank=" << myid
                      << " global_k=" << i_k_point
                      << " spin=" << is
                      << " s_max=" << s_max
                      << " s_min=" << s_min
                      << " rank_tolerance=" << rank_tolerance << std::endl;
            projector_abort("rank-deficient weighted projector", i_k_point, is, 0);
          }

          const double rcond = s_min/s_max;
          if(!std::isfinite(rcond) || rcond < 1.0e-10)
          {
            std::cerr << "[CWF ill-conditioned spectrum]"
                      << " rank=" << myid
                      << " global_k=" << i_k_point
                      << " spin=" << is
                      << " s_max=" << s_max
                      << " s_min=" << s_min
                      << " rcond=" << rcond << std::endl;
            projector_abort("ill-conditioned weighted projector", i_k_point, is, 0);
          }
          cwf_rcond_min_local = std::min(cwf_rcond_min_local, rcond);

          proj_mat_tmp = svd.matrixU() * svd.matrixV().adjoint();
          norm_orth = proj_mat_tmp.adjoint() * proj_mat_tmp;

          double orthogonality_error = 0.0;
          for(int iorb1=0; iorb1<norb; iorb1++)
            for(int iorb2=0; iorb2<norb; iorb2++)
            {
              const std::complex<double> target =
                  iorb1 == iorb2 ? one : zero;
              const double residual =
                  std::abs(norm_orth(iorb1,iorb2) - target);
              if(!std::isfinite(residual))
                projector_abort("non-finite CWF orthogonality residual",
                                i_k_point, is, 0);
              orthogonality_error = std::max(orthogonality_error, residual);
            }

          cwf_orth_max_local =
              std::max(cwf_orth_max_local, orthogonality_error);
          if(orthogonality_error > 1.0e-10)
            projector_abort("CWF projector is not column-orthonormal",
                            i_k_point, is, 0);
        }

        for(int iatom=0; iatom<natom; iatom++)
        {
          const int m_tot = norb_sub[iatom];
          Matrix& P = this->projector_mat[ik][iatom][is];
          for(int m=0; m<m_tot; m++)
            P.col(m) = proj_mat_tmp.col(orb_index[iatom][m]);

          Matrix& H = this->unitary_trans[ik][iatom][is];
          const Eigen::Map<const Eigen::VectorXd> energies(
              epsilon[is][i_k_point].data(), wbands[is]);
          const Matrix weighted_P = energies.asDiagonal() * P;
          H = P.adjoint() * weighted_P;
          for(int m=0; m<m_tot; m++)
            this->TB_Hk[ik][iatom][is][m] = H(m,m);

          if(projector_type == 2)
          {
            double hermiticity_error = 0.0;
            for(int m1=0; m1<m_tot; m1++)
              for(int m2=0; m2<m_tot; m2++)
              {
                const double residual = std::abs(
                    H(m1,m2) - std::conj(H(m2,m1)));
                if(!std::isfinite(residual))
                  projector_abort("non-finite local-Hamiltonian residual",
                                  i_k_point, is, 0);
                hermiticity_error = std::max(hermiticity_error, residual);
              }

            cwf_hermitian_max_local =
                std::max(cwf_hermitian_max_local, hermiticity_error);
            if(hermiticity_error > 1.0e-10)
              projector_abort("CWF local Hamiltonian is not Hermitian",
                              i_k_point, is, 0);
          }
        }

      }//is
    }//ik

    finish_window_overlap();

    // Rank 0 always owns k=0; share its basis size with ranks that may own no
    // k point when more MPI tasks than k points are requested.
    MPI_Bcast(&this->n_basis, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(projector_type == 2)
    {
      double cwf_energy_min = 0.0;
      double cwf_energy_max = 0.0;
      double cwf_window_min = 0.0;
      double cwf_window_max = 0.0;
      double cwf_rcond_min = 0.0;
      double cwf_orth_max = 0.0;
      double cwf_hermitian_max = 0.0;

      MPI_Reduce(&cwf_energy_min_local, &cwf_energy_min, 1,
                 MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&cwf_energy_max_local, &cwf_energy_max, 1,
                 MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&cwf_window_min_local, &cwf_window_min, 1,
                 MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&cwf_window_max_local, &cwf_window_max, 1,
                 MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&cwf_rcond_min_local, &cwf_rcond_min, 1,
                 MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&cwf_orth_max_local, &cwf_orth_max, 1,
                 MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&cwf_hermitian_max_local, &cwf_hermitian_max, 1,
                 MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      if(myid == 0)
      {
        std::ostringstream summary;
        summary << std::scientific << std::setprecision(12)
                << "CWF energy reference: E_KS-E_F (eV)" << std::endl
                << "CWF relative window (eV): [" << CWF_ep0
                << ", " << CWF_ep1 << "]" << std::endl
                << "CWF broadening T (eV): " << CWF_T << std::endl
                << "CWF correlated KS energy range (eV): ["
                << cwf_energy_min << ", " << cwf_energy_max << "]" << std::endl
                << "CWF window-weight range: [" << cwf_window_min
                << ", " << cwf_window_max << "]" << std::endl
                << "CWF minimum singular-value ratio: "
                << cwf_rcond_min << std::endl
                << "CWF maximum |P^dagger P-I|: "
                << cwf_orth_max << std::endl
                << "CWF maximum Hloc Hermiticity residual (Hartree): "
                << cwf_hermitian_max << std::endl;
        GLV::ofs_running << summary.str();
      }
    }

    timer::get_time(time, seconds);
    GLV::ofs_running << "Time consuming for contruction of the projector: " 
                << (int)seconds << "s" << std::endl;

    //=========TEST orthonormality========================
    /*
    std::vector<std::complex<double>> test_mat(norb*norb);
    for(int is=0; is<nspin; is++)
    {
      std::vector<std::complex<double>> norm(norb*norb);
      for(int ik=0; ik<k_map.size(); ik++)
      {
        for(int iatom1=0; iatom1<natom; iatom1++)
        {
          const int m_tot1 = norb_sub[iatom1];
          for(int m1=0; m1<m_tot1; m1++)
          {
            const int iorb1 = orb_index[iatom1][m1];
            for(int iatom2=0; iatom2<natom; iatom2++)
            {
              const int m_tot2 = norb_sub[iatom2];
              for(int m2=0; m2<m_tot2; m2++)
              {
                const int iorb2 = orb_index[iatom2][m2];
                norm[iorb1*norb+iorb2] = zero;
                for(int iband=0; iband<wbands[is]; iband++)
                  norm[iorb1*norb+iorb2] += std::conj(this->projector_mat[ik][iatom1][is][iband*m_tot1+m1])*
                                       this->projector_mat[ik][iatom2][is][iband*m_tot2+m2];
              }
            }//iatom2
          }//m1
        }//iatom1

        std::stringstream ss;
        ss << "projector_orthonormality/norm_ik" << k_map[ik] << ".dat";
        std::ofstream ofs(ss.str().c_str(), std::ios::out);
        // ofs << "process: " << mpi_rank() << std::endl;
        for(int iatom1=0; iatom1<natom; iatom1++)
        {
          const int m_tot1 = norb_sub[iatom1];
          for(int m1=0; m1<m_tot1; m1++)
          {
            const int iorb1 = orb_index[iatom1][m1];
            for(int iatom2=0; iatom2<natom; iatom2++)
            {
              const int m_tot2 = norb_sub[iatom2];
              for(int m2=0; m2<m_tot2; m2++)
              {
                const int iorb2 = orb_index[iatom2][m2];
                ofs << is << std::setw(5) << ik << std::setw(3) << iorb1 << std::setw(3) << iorb2 <<
                std::setw(15) << std::fixed << std::setprecision(9) << norm[iorb1*norb+iorb2].real()
                << std::setw(15) << std::fixed << std::setprecision(9) << norm[iorb1*norb+iorb2].imag() << std::endl;
              }
            }//iatom2
          }//m1
        }//iatom1
        ofs.close();
      }//ik
    }//is
    */

    //=============================================
    //   unitary tranform of local orbital to
    //   get diagonal Green function, hybridization
    //   function and etc.
    //   Tight binding Hamiltonian.
    //=============================================
    if(atom.local_sym()==0 || atom.local_sym()==1 || atom.local_sym()==2 || atom.local_sym()==3) return;
    auto projector_mat_tmp = this->projector_mat;

    for(int ineq=0; ineq<atom.inequ_atoms(); ineq++)
    {
      const int iatom = atom.ineq_iatom(ineq);
      const int m_tot = norb_sub[iatom];

      for(int ik=0; ik<k_map.size(); ik++)
      {
        for(int is=0; is<nspin; is++)
        {
          Matrix& rotation = this->unitary_trans[ik][iatom][is];
          Eigen::SelfAdjointEigenSolver<Matrix> solver;
          solver.compute(rotation.selfadjointView<Eigen::Upper>());
          if(solver.info() != Eigen::Success)
            projector_abort("local-Hamiltonian eigensolver failed",
                            k_map[ik], is, solver.info());
          rotation = solver.eigenvectors();
          for(int m=0; m<m_tot; m++)
            this->TB_Hk[ik][iatom][is][m] = solver.eigenvalues()[m];

          const Matrix& P0 = projector_mat_tmp[ik][iatom][is];
          Matrix& P = this->projector_mat[ik][iatom][is];
          P = P0 * rotation;

          for(int iatom1=0; iatom1<natom; iatom1++)
            if(iatom1!=iatom && atom.equ_atom(iatom1)==ineq){
              const Matrix& P_equiv0 = projector_mat_tmp[ik][iatom1][is];
              Matrix& P_equiv = this->projector_mat[ik][iatom1][is];
              P_equiv = P_equiv0 * rotation;
            }

        }//is
      }//ik  
    }//iatom

//=========TEST orthonormality========================
// for(int is=0; is<nspin; is++)
// {
//   std::vector<std::complex<double>> norm(norb*norb);
//   int ik_count=-1;
//   for(int ik=0; ik<nkpoints; ik++)
//   {
//     if(ik%mpi_ntasks() != mpi_rank()) continue;  //k_points are divided acording to process id
//     ik_count++;
//     for(int iatom1=0; iatom1<natom; iatom1++)
//     {
//       const int m_tot1 = norb_sub[iatom1];
//       for(int m1=0; m1<m_tot1; m1++)
//       {
//         const int iorb1 = orb_index[iatom1][m1];
//         for(int iatom2=0; iatom2<natom; iatom2++)
//         {
//           const int m_tot2 = norb_sub[iatom2];
//           for(int m2=0; m2<m_tot2; m2++)
//           {
//             const int iorb2 = orb_index[iatom2][m2];
//             norm[iorb1*norb+iorb2] = zero;
//             for(int iband=0; iband<wbands[is]; iband++)
//               norm[iorb1*norb+iorb2] += std::conj(this->projector_mat[ik_count][iatom1][is][iband*m_tot1+m1])*
//                                    this->projector_mat[ik_count][iatom2][is][iband*m_tot2+m2];
//           }
//         }//iatom2
//       }//m1
//     }//iatom1

//     std::stringstream ss;
//     ss << "projector_orthonormality/norm_ik" << ik << ".dat";
//     std::ofstream ofs(ss.str().c_str(), std::ios::out);

//     for(int iatom1=0; iatom1<natom; iatom1++)
//     {
//       const int m_tot1 = norb_sub[iatom1];
//       for(int m1=0; m1<m_tot1; m1++)
//       {
//         const int iorb1 = orb_index[iatom1][m1];
//         for(int iatom2=0; iatom2<natom; iatom2++)
//         {
//           const int m_tot2 = norb_sub[iatom2];
//           for(int m2=0; m2<m_tot2; m2++)
//           {
//             const int iorb2 = orb_index[iatom2][m2];
//             ofs << is << std::setw(5) << ik << std::setw(3) << iorb1 << std::setw(3) << iorb2 <<
//             std::setw(15) << std::fixed << std::setprecision(9) << norm[iorb1*norb+iorb2].real()
//             << std::setw(15) << std::fixed << std::setprecision(9) << norm[iorb1*norb+iorb2].imag() << std::endl;
//           }
//         }//iatom2
//       }//m1
//     }//iatom1
//     ofs.close();
//   }//ik
// }//is


// //=========TEST completeness========================
// for(int is=0; is<nspin; is++)
// {
//   std::vector<std::complex<double>> norm(wbands[is]*wbands[is]);
//   int ik_count=-1;
//   for(int ik=0; ik<nkpoints; ik++)
//   {
//     if(ik%mpi_ntasks() != mpi_rank()) continue;  //k_points are divided acording to process id
//     ik_count++;

//     for(int iband1=0; iband1<wbands[is]; iband1++)
//     {
//       for(int iband2=0; iband2<wbands[is]; iband2++)
//       {
//         norm[iband1*wbands[is]+iband2] = zero;
//         for(int iorb=0; iorb<norb; iorb++)
//           norm[iband1*wbands[is]+iband2] += std::conj(this->projector_mat[ik_count][iband2*norb+iorb])*
//                                this->projector_mat[ik_count][is][iband1*norb+iorb];
//       }
//     }//m

//     std::stringstream ss;
//     ss << "projector_completeness/norm_ik" << ik << ".dat";
//     std::ofstream ofs(ss.str().c_str(), std::ios::out);

//     for(int iband1=0; iband1<wbands[is]; iband1++)
//     {
//       for(int iband2=0; iband2<wbands[is]; iband2++)
//       {
//         ofs << is << std::setw(5) << ik << std::setw(3) << iband1 << std::setw(3) << iband2 <<
//         std::setw(15) << std::fixed << std::setprecision(9) << norm[iband1*wbands[is]+iband2].real()
//         << std::setw(15) << std::fixed << std::setprecision(9) << norm[iband1*wbands[is]+iband2].imag() << std::endl;
//       }
//     }
//     ofs.close();
//   }//ik
// }//is

// DFT_output::overlap_matrix ovlp1(DFT_solver,"../DFT/outputs/overlap_matrix");
// wave_function wfc1;

// int ik_count_tmp=-1;
// for(int ik=0; ik<nkpoints; ik++)
// {
//   if(ik%mpi.ntasks() != mpi.rank()) continue;  //k_points are divided acording to process id
//   ik_count_tmp++;
  
//   ovlp1.evaluate_ovlp_k(ik, atom);   //caculate overlap matrix in k-space
//   this->evalute_projector_k(ovlp1, orth, wfc1, atom, ik, ik_count_tmp);   
// }

    return;
  }

}
