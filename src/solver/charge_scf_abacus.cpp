#include "charge_scf_abacus.h"
#include "density_matrix.h"
#include "solver.h"
#include "lattice.h"
#include "../para/KS_eigenvectors.h"
#include "../mpi_environment.h"
#include "../global_variables.h"
#include "../constants.h"
#include <mpi.h>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <sys/stat.h>

namespace DFT_plus_DMFT
{
namespace
{

    // ABACUS real CSR; empty blocks have no value/column/pointer lines.
    void read_csr(const std::string& name, bool step_header,
                  std::vector<std::array<int,3>>& R, std::vector<RealMatrix>& matrices)
    {
        std::ifstream file(name);
        std::string line, content;
        while(std::getline(file,line)){
            const auto comment=line.find('#');
            if(comment!=std::string::npos) line.erase(comment);
            if(line.find_first_not_of(" \t\r")==std::string::npos) continue;
            content+=line+"\n";
        }
        std::istringstream in(content);
        if(step_header) std::getline(in,line);
        std::getline(in,line);
        const int n=std::stoi(line.substr(line.find_last_of(' ')+1));
        std::getline(in,line);
        const int nr=std::stoi(line.substr(line.find_last_of(' ')+1));
        R.clear(); matrices.clear();
        for(int ir=0;ir<nr;ir++){
            std::array<int,3> r;
            int count;
            in>>r[0]>>r[1]>>r[2]>>count;
            RealMatrix block=RealMatrix::Zero(n,n);
            if(count){
                std::vector<double> values(count);
                std::vector<int> columns(count),pointers(n+1);
                for(auto& value:values) in>>value;
                for(auto& col:columns) in>>col;
                for(auto& ptr:pointers) in>>ptr;
                for(int i=0;i<n;i++)
                    for(int p=pointers[i];p<pointers[i+1];p++) block(i,columns[p])=values[p];
            }
            R.push_back(r); matrices.push_back(block);
        }
        if(!in) throw std::runtime_error("Incomplete CSR: "+name);
    }

    std::vector<double> read_cube(const std::string& name)
    {
        std::ifstream in(name);
        std::string line;
        std::getline(in,line); std::getline(in,line);
        int natom,n[3]; double x,y,z;
        in>>natom>>x>>y>>z;
        for(int i=0;i<3;i++) in>>n[i]>>x>>y>>z;
        for(int i=0;i<natom;i++){int atom;double charge;in>>atom>>charge>>x>>y>>z;}
        std::vector<double> rho(n[0]*n[1]*n[2]);
        for(auto& value:rho) in>>value;
        if(!in) throw std::runtime_error("Incomplete cube: "+name);
        return rho;
    }
}

void Charge_SCF_abacus::init()
{
    read_csr("abacus_charge_reference.csr",true,R4dm,DMR_in);
    rho_in=read_cube("abacus_charge_reference.cube");
    // Cube grid vectors are rounded; use the exact STRU cell volume in bohr^3.
    std::ifstream volume_file("abacus_charge_volume.dat");
    double volume;
    if(!(volume_file>>volume) || !(volume>0))
        throw std::runtime_error("Missing valid abacus_charge_volume.dat (bohr^3)");
    dv=volume/rho_in.size();
    if(mpi_rank()==0) mkdir("abacus_charge",0755);
    MPI_Barrier(MPI_COMM_WORLD);
}

bool Charge_SCF_abacus::update(solver& task, int charge_step)
{
    auto& in=task.pars.in;
    auto& band=task.pars.bands;
    auto& atom=task.pars.atom;
    if(band.soc() || band.nspins()!=1 || *(int*)in.parameter("magnetism")!=3)
        throw std::runtime_error("ABACUS charge backend currently requires paramagnetic non-SOC nspin=1");
    const double beta=*(double*)in.parameter("beta");
    const double mixing=*(double*)in.parameter("charge_mix_param");
    task.imp.sigma.subtract_double_counting(0);
    const auto& local=task.imp.sigma.correlated_sigma(0);
    const auto& window=task.space.wbands2ibands()[0];
    const int natom=atom.total_atoms();
    const int n_omega=local[0][0].size();
    const int n_band_window=window.size();
    auto Sigma_local=std::vector<std::vector<Matrix>>(natom,std::vector<Matrix>(n_omega));
    for(int a=0;a<natom;a++){
        const int norb=atom.iatom_norb()[a];
        for(int iw=0;iw<n_omega;iw++)
            Sigma_local[a][iw]=local[atom.equ_atom(a)][0][iw];
    }
    std::vector<int> k_indices;
    std::vector<std::vector<Matrix>> Sigma_k;
    for(int ik=0;ik<band.nk();ik++){
        if(ik%mpi_ntasks()!=mpi_rank()) continue;
        auto P=std::vector<Matrix>(natom);
        for(int a=0;a<natom;a++)
            P[a]=task.proj.proj_access(k_indices.size())[a][0];
        k_indices.push_back(ik);
        Sigma_k.push_back(lattice::self_energy(n_band_window,n_omega,P,Sigma_local));
    }
    auto number = [&](double mu){
        double partial=0,total=0;
        for(int j=0;j<int(k_indices.size());j++){
            int ik=k_indices[j];
            auto DM_window=density_matrix::density_matrix_window(task.space.eigen_val()[0][ik],Sigma_k[j],beta,mu);
            partial+=2*band.kweight()[ik]*density_matrix::electron_number(band.enk()[0][ik],window,DM_window,beta,mu);
        }
        MPI_Allreduce(&partial,&total,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        return total;
    };
    double mu=task.Mu.mu_corrected(), lower=mu-1,upper=mu+1;
    if(number(lower)>band.n_electrons() || number(upper)<band.n_electrons())
        throw std::runtime_error("ABACUS density electron-number root not bracketed");
    double electrons=0;
    for(int i=0;i<60;i++){
        mu=(lower+upper)/2; electrons=number(mu);
        if(std::abs(electrons-band.n_electrons())<1e-8) break;
        if(electrons>band.n_electrons()) upper=mu; else lower=mu;
    }
    if(std::abs(electrons-band.n_electrons())>=1e-8)
        throw std::runtime_error("ABACUS density electron number not converged");
    std::ifstream kfile("dft/outputs_to_DMFT/overlap_matrix/k_vector.dat");
    int nk,index; kfile>>nk;
    std::vector<std::array<double,3>> k(nk);
    for(int ik=0;ik<nk;ik++) kfile>>index>>k[ik][0]>>k[ik][1]>>k[ik][2];
    if(!kfile || nk!=band.nk()) throw std::runtime_error("Invalid ABACUS k mesh");
    std::vector<std::array<int,3>> R4ovlp;
    std::vector<RealMatrix> ovlp_mat;
    read_csr("dft/outputs_to_DMFT/overlap_matrix/SR.csr",false,R4ovlp,ovlp_mat);
    auto DMR_out=std::vector<RealMatrix>(R4dm.size(),RealMatrix::Zero(DMR_in[0].rows(),DMR_in[0].cols()));
    double trace_partial=0,trace_total=0;
    for(int j=0;j<int(k_indices.size());j++){
        int ik=k_indices[j];
        auto DM_window=density_matrix::density_matrix_window(task.space.eigen_val()[0][ik],Sigma_k[j],beta,mu);
        DFT_output::KS_eigenvectors wfc;
        std::vector<Matrix> C;
        wfc.read_eigenvec("dft/outputs_to_DMFT/KS_eigenvector/",ik,C);
        const Matrix& Cm=C[0];
        Matrix DMK=density_matrix::density_matrix_NAO(Cm,band.enk()[0][ik],window,DM_window,beta,mu);
        Matrix S=Matrix::Zero(wfc.basis_n(),wfc.basis_n());
        for(int ir=0;ir<int(R4ovlp.size());ir++){
            double phase=2*GLC::PI*(k[ik][0]*R4ovlp[ir][0]+k[ik][1]*R4ovlp[ir][1]+k[ik][2]*R4ovlp[ir][2]);
            S+=std::exp(std::complex<double>(0,phase))*ovlp_mat[ir];
        }
        trace_partial+=2*band.kweight()[ik]*(DMK*S).trace().real();
        auto contribution=density_matrix::density_matrix_R({DMK},{k[ik]},{band.kweight()[ik]},R4dm,2.0);
        for(int ir=0;ir<int(R4dm.size());ir++) DMR_out[ir]+=contribution[ir];
    }
    MPI_Allreduce(&trace_partial,&trace_total,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    if(std::abs(trace_total-band.n_electrons())>1e-6)
        throw std::runtime_error("ABACUS density trace(DS) has incorrect electron number");
    for(auto& block:DMR_out) MPI_Allreduce(MPI_IN_PLACE,block.data(),block.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    const std::string folder="abacus_charge/step"+std::to_string(charge_step);
    if(mpi_rank()==0){
        if(mkdir(folder.c_str(),0755)!=0) throw std::runtime_error("Charge step directory already exists: "+folder);
        density_matrix::write_density_matrix_R(folder+"/dmft.csr",R4dm,DMR_out);
    }
    for(int ir=0;ir<int(R4dm.size());ir++) DMR_in[ir]+=mixing*(DMR_out[ir]-DMR_in[ir]);
    int status=0;
    if(mpi_rank()==0){
        density_matrix::write_density_matrix_R(folder+"/dmrs1_nao.csr",R4dm,DMR_in);
        status=std::system(("bash ./abacus_update.sh "+std::to_string(charge_step)).c_str());
    }
    MPI_Bcast(&status,1,MPI_INT,0,MPI_COMM_WORLD);
    if(status!=0) throw std::runtime_error("ABACUS single-step bridge failed; inspect "+folder);
    auto rho=read_cube(folder+"/charge.cube");
    double squared=0,grid_number=0;
    for(int i=0;i<int(rho.size());i++){
        squared+=std::pow((rho[i]-rho_in[i])/mixing,2);
        grid_number+=rho[i]*dv;
    }
    double residual=std::sqrt(dv*squared);
    rho_in=std::move(rho);
    if(mpi_rank()==0){
        std::ofstream report(folder+"/summary.dat");
        report<<std::setprecision(17)<<"mu_Hartree "<<mu<<"\nelectrons "<<electrons
              <<"\ntrace_DS "<<trace_total<<"\ngrid_electrons "<<grid_number<<"\ndensity_residual "<<residual<<"\n";
        GLV::ofs_running<<"ABACUS charge step "<<charge_step<<": density residual = "<<residual
                        <<", electrons = "<<grid_number<<", density mu (Hartree) = "<<mu<<std::endl;
    }
    return residual<*(double*)in.parameter("delta_rho");
}
}
