#include "comctqmc.h"
#include "../constants.h"
#include "../mpi_environment.h"
#include "../global_variables.h"
#include "../../thirdparty/ComCTQMC/include/io/Vector.h"
#include <mpi.h>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <stdexcept>
#include <sys/stat.h>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <unistd.h>

namespace
{
    std::string impurity_directory(int charge_step, int dmft_step, int q)
    {
        return "dmft/charge_step"+std::to_string(charge_step)+"/dmft_step"
            +std::to_string(dmft_step)+"/impurity"+std::to_string(q)+"/comctqmc";
    }

    std::string shell_quote(const std::string& value)
    {
        std::string result="'";
        for(char c:value) result+=c=='\'' ? "'\\''" : std::string(1,c);
        return result+"'";
    }

    std::string executable_path(const std::string& value)
    {
        if(value.empty()) throw std::runtime_error("Empty ComCTQMC executable path");
        if(value[0]=='/' || value.find('/')==std::string::npos) return value;
        char* directory=getcwd(nullptr,0);
        if(!directory) throw std::runtime_error("Cannot resolve ComCTQMC executable path");
        std::string path=std::string(directory)+"/"+value;
        std::free(directory);
        return path;
    }
}

using DFT_plus_DMFT::Matrix;

namespace DMFT
{
    void ComCTQMC::write_input(const std::string& directory,double beta,double mu,
        const std::vector<double>& levels,
        const std::vector<std::vector<std::complex<double>>>& delta,
        const interaction_tensor& U,long long thermalisation_steps,long long measurement_steps)
    {
        const int L=levels.size();
        const int N=delta[0].size();
        const double unit=GLC::Hartree_to_eV;
        double average=0.0;
        for(double e:levels) average+=e/L;
        std::ofstream params(directory+"/params.json"), bath(directory+"/hyb.json");
        if(!params || !bath) throw std::runtime_error("Cannot write ComCTQMC input in "+directory);
        params.imbue(std::locale::classic()); bath.imbue(std::locale::classic());
        params << std::setprecision(17); bath << std::setprecision(17);
        params << "{\n  \"beta\": " << beta/unit << ",\n  \"mu\": " << (mu-average)*unit
            << ",\n  \"basis\": {\"orbitals\": " << L/2 << "},\n  \"hloc\": {\n    \"one body\": [\n";
        for(int a=0;a<L;a++)
        {
            params << "      [";
            for(int b=0;b<L;b++) params << (b ? ", " : "") << (a==b ? (levels[a]-average)*unit : 0.0);
            params << "]" << (a+1<L ? "," : "") << '\n';
        }
        params << "    ],\n    \"two body\": [";
        // The solver divides this raw tensor by 2 internally. Flatten i,j,k,l.
        for(int a=0;a<L;a++) for(int b=0;b<L;b++) for(int c=0;c<L;c++) for(int d=0;d<L;d++)
            params << ((a || b || c || d) ? ", " : "") << U[a][b][c][d]*unit;
        params << "]\n  },\n  \"hybridisation\": {\n    \"functions\": \"hyb.json\",\n    \"matrix\": [\n";
        for(int a=0;a<L;a++)
        {
            params << "      [";
            for(int b=0;b<L;b++)
            {
                params << (b ? ", " : "") << '"';
                if(a==b) params << "f" << a;
                params << '"';
            }
            params << "]" << (a+1<L ? "," : "") << '\n';
        }
        params << "    ]\n  },\n  \"observables\": {},\n  \"thermalisation steps\": " << thermalisation_steps
            << ",\n  \"measurement steps\": " << measurement_steps
            << ",\n  \"green bulla\": true,\n  \"green basis\": \"legendre\",\n"
            << "  \"green legendre cutoff\": 100,\n  \"green matsubara cutoff\": " << 2*GLC::PI*(N+0.5)/(beta/unit)
            << ",\n  \"density matrix precise\": true\n}\n";
        bath << "{\n";
        for(int a=0;a<L;a++)
        {
            bath << "  \"f" << a << "\": {\"real\": [";
            for(int n=0;n<N;n++) bath << (n ? ", " : "") << delta[a][n].real()*unit;
            bath << "], \"imag\": [";
            for(int n=0;n<N;n++) bath << (n ? ", " : "") << delta[a][n].imag()*unit;
            bath << "]}" << (a+1<L ? "," : "") << '\n';
        }
        bath << "}\n";
        params.flush(); bath.flush();
        if(!params || !bath) throw std::runtime_error("Failed writing ComCTQMC input");
    }

    void ComCTQMC::output(int charge_step,int dmft_step,double mu,input_info& in,
        DFT_output::atoms_info& atom,DFT_output::KS_bands& band,
        const std::vector<std::vector<DFT_plus_DMFT::Matrix>>& Eimp,
        const std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>& hyb,
        coulomb_tensor& Umat)
    {
        executable=*(const std::string*)in.parameter("comctqmc_exe");
        evalsim=*(const std::string*)in.parameter("comctqmc_evalsim");
        launcher=*(const std::string*)in.parameter("comctqmc_launcher");
        nproc=*(int*)in.parameter("comctqmc_nproc");
        const int nspin=band.nspins();
        if(band.soc() || (nspin!=1 && nspin!=2))
            throw std::runtime_error("ComCTQMC input currently requires collinear spins without SOC");
        const std::vector<std::string> paths={"dmft", "dmft/charge_step"+std::to_string(charge_step),
            "dmft/charge_step"+std::to_string(charge_step)+"/dmft_step"+std::to_string(dmft_step)};
        for(const auto& path:paths)
            if(mkdir(path.c_str(),0755)!=0 && errno!=EEXIST)
                throw std::runtime_error("Cannot create "+path);
        for(int q=0;q<atom.inequ_atoms();q++)
        {
            const int M=atom.iatom_norb()[atom.ineq_iatom(q)];
            const int N=hyb[q][0].size();
            std::vector<double> levels(2*M);
            std::vector<std::vector<std::complex<double>>> delta(2*M,std::vector<std::complex<double>>(N));
            for(int s=0;s<2;s++) for(int m=0;m<M;m++)
            {
                int spin=nspin==1 ? 0 : s;
                levels[s*M+m]=Eimp[q][spin](m, m).real();
                for(int n=0;n<N;n++) delta[s*M+m][n]=hyb[q][spin][n](m, m);
            }
            const std::string directory=paths.back()+"/impurity"+std::to_string(q)+"/comctqmc";
            const std::string impurity=paths.back()+"/impurity"+std::to_string(q);
            if(mkdir(impurity.c_str(),0755)!=0 && errno!=EEXIST) throw std::runtime_error("Cannot create "+impurity);
            if(mkdir(directory.c_str(),0755)!=0 && errno!=EEXIST) throw std::runtime_error("Cannot create "+directory);
            write_input(directory,*(double*)in.parameter("beta"),mu,levels,delta,
                Umat.Coulomb_matrix()[q],*(long long*)in.parameter("warmup"),
                *(long long*)in.parameter("mc_step"));
        }
    }

    void ComCTQMC::run(const std::string& directory,const std::string& executable,
        const std::string& evalsim,const std::string& launcher,int nproc)
    {
        if(nproc<0) throw std::runtime_error("Invalid ComCTQMC MPI process count");
        const std::string command="cd "+shell_quote(directory)+" && "
            +shell_quote(executable_path(launcher))+" "+std::to_string(nproc)+" "
            +shell_quote(executable_path(executable))+" "+shell_quote(executable_path(evalsim))
            +" > launch.log 2>&1";
        if(std::system(command.c_str())!=0)
            throw std::runtime_error("ComCTQMC failed in "+directory+"; inspect launch.log, ctqmc.log and evalsim.log");
    }

    void ComCTQMC::read_functions(const std::string& directory,double beta,int L,int N,
        std::vector<std::vector<std::complex<double>>>& G,
        std::vector<std::vector<std::complex<double>>>& S)
    {
        jsx::value params,obs;
        std::ifstream input(directory+"/params.json"),output(directory+"/params.obs.json");
        if(!input || !output) throw std::runtime_error("Cannot read ComCTQMC results in "+directory);
        jsx::read(input,params); jsx::read(output,obs);
        const double unit=GLC::Hartree_to_eV;
        const double input_beta=params("beta").real64();
        if(std::abs(input_beta-beta/unit)>1e-10*std::abs(beta/unit)
            || params("basis")("orbitals").int64()*2!=L)
            throw std::runtime_error("ComCTQMC result mesh or orbital count does not match DMFT");
        G.resize(L); S.resize(L);
        for(int a=0;a<L;a++)
        {
            const std::string name="f"+std::to_string(a);
            const auto g=jsx::at<io::cvec>(obs("green")(name)("function"));
            const auto s=jsx::at<io::cvec>(obs("self-energy")(name)("function"));
            if(int(g.size())!=N || int(s.size())!=N)
                throw std::runtime_error("Wrong number of ComCTQMC Matsubara frequencies");
            G[a].resize(N); S[a].resize(N);
            for(int n=0;n<N;n++)
            {
                if(!std::isfinite(g[n].real()) || !std::isfinite(g[n].imag())
                    || !std::isfinite(s[n].real()) || !std::isfinite(s[n].imag()))
                    throw std::runtime_error("Nonfinite ComCTQMC result");
                G[a][n]=g[n]*unit;
                S[a][n]=s[n]/unit;
            }
        }
    }

    void ComCTQMC::impurities_solving(int charge_step,int dmft_step,
        DFT_output::atoms_info& atom)
    {
        for(int q=0;q<atom.inequ_atoms();q++)
        {
            int status=0;
            if(mpi_rank()==0)
            {
                try {
                    const std::string directory=impurity_directory(charge_step,dmft_step,q);
                    GLV::ofs_running << "  ComCTQMC impurity" << q << " starts" << std::endl;
                    run(directory,executable,evalsim,launcher,nproc);
                    GLV::ofs_running << "  ComCTQMC impurity" << q << " finished" << std::endl;
                }
                catch(const std::exception& error) {
                    std::cerr << error.what() << std::endl;
                    status=1;
                }
            }
            MPI_Request request;
            MPI_Ibcast(&status,1,MPI_INT,0,MPI_COMM_WORLD,&request);
            int finished=0;
            while(!finished)
            {
                MPI_Test(&request,&finished,MPI_STATUS_IGNORE);
                if(!finished) usleep(10000);
            }
            if(status) MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
        }
    }

    void ComCTQMC::read_self_energy(int charge_step,int dmft_step,DFT_output::KS_bands& band,
        input_info& in,DFT_output::atoms_info& atom,
        std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>& S)
    {
        try {
            const int nspin=band.nspins(),N=*(int*)in.parameter("n_omega");
            const int magnetism=*(int*)in.parameter("magnetism");
            const double beta=*(double*)in.parameter("beta");
            if(band.soc() || (nspin!=1 && nspin!=2))
                throw std::runtime_error("ComCTQMC results require collinear spins without SOC");
            for(int q=0;q<atom.inequ_atoms();q++)
            {
                const int iatom=atom.ineq_iatom(q),M=atom.iatom_norb()[iatom];
                std::vector<std::vector<std::complex<double>>> g,s;
                read_functions(impurity_directory(charge_step,dmft_step,q),beta,2*M,N,g,s);
                for(int spin=0;spin<nspin;spin++) for(int n=0;n<N;n++)
                {
                    std::vector<std::complex<double>> sd(M);
                    for(int m=0;m<M;m++)
                    {
                        if(nspin==1 || magnetism==3 || magnetism==4)
                        {
                            sd[m]=(s[m][n]+s[M+m][n])/2.0;
                        }
                        else sd[m]=s[spin*M+m][n];
                    }
                    DFT_output::atoms_info::symmetry_operation_vector(atom.local_sym(),atom.L(iatom),M,sd.data());
                    S[q][spin][n].setZero();
                    for(int m=0;m<M;m++)
                    {
                        S[q][spin][n](m, m)=sd[m];
                    }
                }
            }
        }
        catch(const std::exception& error) {
            std::cerr << "ComCTQMC result: " << error.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
        }
    }
}
