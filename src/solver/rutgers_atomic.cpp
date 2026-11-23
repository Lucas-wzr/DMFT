#include "rutgers_atomic.h"
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>


namespace DMFT
{
  namespace rutgers_atomic
  {
    namespace
    {
      int occupation(int q)
      {
        int N = 0;
        for(; q; q >>= 1) N += q & 1;
        return N;
      }

      double spin(int q, int M)
      {
        return 0.5*(occupation(q & ((1 << M)-1))-occupation(q >> M));
      }

      bool apply(int& q, int a, bool create, int& sign)
      {
        if(((q >> a) & 1) == int(create)) return false;
        if(occupation(q & ((1 << a)-1)) % 2) sign = -sign;
        q ^= 1 << a;
        return true;
      }

      int root(std::vector<int>& parent, int q)
      {
        if(parent[q] != q) parent[q] = root(parent, parent[q]);
        return parent[q];
      }

      bool merge(std::vector<int>& parent, int a, int b)
      {
        a = root(parent, a);
        b = root(parent, b);
        if(a == b) return false;
        parent[std::max(a,b)] = std::min(a,b);
        return true;
      }
    }

    interaction_tensor read_tensor(std::istream& input, int M, double energy_unit)
    {
      const int L = 2*M;
      interaction_tensor U(L, std::vector<std::vector<std::vector<double>>>(L,
               std::vector<std::vector<double>>(L, std::vector<double>(L, 0.0))));
      std::set<std::vector<int>> entries;
      std::string line;
      while(std::getline(input, line))
      {
        const std::size_t comment = line.find('#');
        if(comment != std::string::npos) line.erase(comment);
        std::istringstream row(line);
        row >> std::ws;
        if(row.eof()) continue;
        int i, j, k, l;
        double value;
        if(!(row >> i >> j >> k >> l >> value))
          throw std::runtime_error("Uijkl: expected i j k l value on each data line");
        std::string extra;
        if(row >> extra) throw std::runtime_error("Uijkl: unexpected extra column");
        if(i<0 || i>=L || j<0 || j>=L || k<0 || k>=L || l<0 || l>=L || !std::isfinite(value))
          throw std::runtime_error("Uijkl: invalid index or nonfinite value");
        if(!entries.insert({i,j,k,l}).second)
          throw std::runtime_error("Uijkl: duplicate tensor entry");
        U[i][j][k][l] = value/energy_unit;
      }
      if(entries.empty()) throw std::runtime_error("Uijkl: no tensor entries found");
      return U;
    }

    std::vector<block> solve(const std::vector<double>& levels, const interaction_tensor& U)
    {
      const int L = levels.size(), M = L/2;
      if(L<2 || L%2 || L>14)
        throw std::runtime_error("Rutgers atomic problem requires 1 to 7 spatial orbitals");
      if(int(U.size()) != L) throw std::runtime_error("Uijkl: incompatible tensor dimensions");
      double scale = 1.0;
      for(double E : levels)
      {
        if(!std::isfinite(E)) throw std::runtime_error("Nonfinite impurity level");
        scale = std::max(scale, std::abs(E));
      }
      struct term { int i, j, k, l; double value; };
      std::vector<term> terms;
      for(int i=0; i<L; i++)
      {
        if(int(U[i].size()) != L) throw std::runtime_error("Uijkl: incompatible tensor dimensions");
        for(int j=0; j<L; j++)
        {
          if(int(U[i][j].size()) != L) throw std::runtime_error("Uijkl: incompatible tensor dimensions");
          for(int k=0; k<L; k++)
          {
            if(int(U[i][j][k].size()) != L) throw std::runtime_error("Uijkl: incompatible tensor dimensions");
            for(int l=0; l<L; l++)
            {
              const double value = U[i][j][k][l];
              if(!std::isfinite(value)) throw std::runtime_error("Nonfinite Uijkl entry");
              scale = std::max(scale, std::abs(value));
              if(value != 0.0 && i != j && k != l) terms.push_back({i,j,k,l,0.5*value});
            }
          }
        }
      }
      const double tol = 1.0e-12*scale;
      // Both symmetries are needed by the raw tensor used in Rutgers HB2 contractions.
      for(int i=0; i<L; i++)
        for(int j=0; j<L; j++)
          for(int k=0; k<L; k++)
            for(int l=0; l<L; l++)
              if(std::abs(U[i][j][k][l]-U[j][i][l][k]) > tol ||
                 std::abs(U[i][j][k][l]-U[l][k][j][i]) > tol)
                throw std::runtime_error("Uijkl must obey U_ijkl=U_jilk=U_lkji for this real-tensor interface");

      const int dim = 1 << L;
      std::vector<std::map<int,double>> H(dim);
      std::vector<int> parent(dim);
      for(int q=0; q<dim; q++) parent[q] = q;
      for(int q=0; q<dim; q++)
      {
        for(int a=0; a<L; a++) H[q][q] += levels[a]*((q >> a) & 1);
        for(const auto& t : terms)
        {
          int p = q, sign = 1;
          if(apply(p,t.l,false,sign) && apply(p,t.k,false,sign) &&
             apply(p,t.j,true,sign) && apply(p,t.i,true,sign))
          {
            if(spin(p,M) != spin(q,M))
              throw std::runtime_error("The current collinear Rutgers interface requires conserved Sz");
            H[q][p] += t.value*sign;
            // Keep individual interaction terms within a block as well as their sum:
            // Rutgers self-energy sampling applies the raw cubic operators separately.
            merge(parent,p,q);
          }
        }
      }
      for(int q=0; q<dim; q++)
        for(const auto& entry : H[q])
        {
          const int p = entry.first;
          const auto reverse = H[p].find(q);
          const double value = reverse == H[p].end() ? 0.0 : reverse->second;
          if(std::abs(entry.second-value) > tol)
            throw std::runtime_error("Atomic Hamiltonian is not Hermitian");
          if(std::abs(entry.second) > tol && p != q)
          {
            if(spin(p,M) != spin(q,M))
              throw std::runtime_error("The current collinear Rutgers interface requires conserved Sz");
            merge(parent,p,q);
          }
        }

      // CIX requires one target block per source block and flavor, in BOTH directions.
      // Merge occupation-state images until creation and annihilation maps are unique.
      bool changed = true;
      while(changed)
      {
        changed = false;
        std::map<int,std::vector<int>> groups;
        for(int q=0; q<dim; q++) groups[root(parent,q)].push_back(q);
        for(const auto& group : groups)
          for(int a=0; a<L; a++)
            for(int create=0; create<2; create++)
            {
              int first = -1;
              for(int q : group.second)
              {
                int p = q, sign = 1;
                if(!apply(p,a,create,sign)) continue;
                if(first < 0) first = p;
                else changed = merge(parent,first,p) || changed;
              }
            }
      }
      std::map<int,int> indices;
      std::vector<block> blocks;
      std::vector<int> block_index(dim), state_index(dim);
      for(int q=0; q<dim; q++)
      {
        const int r = root(parent,q);
        if(!indices.count(r))
        {
          indices[r] = blocks.size();
          blocks.push_back(block());
        }
        const int b = indices[r];
        block_index[q] = b;
        state_index[q] = blocks[b].states.size();
        blocks[b].states.push_back(q);
      }
      for(auto& b : blocks)
      {
        const int size = b.states.size();
        b.N = occupation(b.states[0]);
        b.Sz = spin(b.states[0],M);
        b.energy.resize(size);
        b.eigenvectors.setZero(size, size);
        DFT_plus_DMFT::RealMatrix matrix = DFT_plus_DMFT::RealMatrix::Zero(size, size);
        for(int col=0; col<size; col++)
          for(const auto& entry : H[b.states[col]])
            if(block_index[entry.first] == block_index[b.states[0]])
              matrix(state_index[entry.first], col) = entry.second;
        if(size == 1)
        {
          b.energy[0] = matrix(0,0);
          b.eigenvectors(0,0) = 1.0;
        }
        else
        {
          Eigen::SelfAdjointEigenSolver<DFT_plus_DMFT::RealMatrix> eig;
          eig.compute(matrix.selfadjointView<Eigen::Upper>());
          if(eig.info() != Eigen::Success)
            throw std::runtime_error("Atomic eigensolver did not converge");
          b.energy = eig.eigenvalues();
          b.eigenvectors = eig.eigenvectors();

        }
        b.target.assign(L,-1);
        b.creation.resize(L);
      }
      for(auto& b : blocks)
      {
        const int size = b.states.size();
        for(int a=0; a<L; a++)
        {
          for(int q : b.states)
          {
            int p = q, sign = 1;
            if(apply(p,a,true,sign)) b.target[a] = block_index[p];
          }
          if(b.target[a] < 0) continue;
          const auto& target = blocks[b.target[a]];
          const int target_size = target.states.size();
          auto& F = b.creation[a];
          F.setZero(size, target_size);
          for(int row=0; row<size; row++)
          {
            int p = b.states[row], sign = 1;
            if(!apply(p,a,true,sign)) continue;
            const int target_row = state_index[p];
            for(int i=0; i<size; i++)
              for(int j=0; j<target_size; j++)
                F(i,j) += sign*b.eigenvectors(row,i)*target.eigenvectors(target_row,j);
          }
        }
      }
      return blocks;
    }

    void write_cix(std::ostream& out, const std::vector<double>& levels,
                   const interaction_tensor& U, const std::vector<block>& blocks, double energy_unit)
    {
      const int L = levels.size(), M = L/2;
      int max_size = 1;
      for(const auto& b : blocks) max_size = std::max(max_size,int(b.states.size()));
      out << std::setprecision(15) << std::scientific;
      out << "# CIX file for ctqmc!\n# cluster_size, number of blocks, number of baths, maximum_matrix_size\n";
      out << "1 " << blocks.size() << ' ' << L << ' ' << max_size << '\n';
      out << "# baths, dimension, symmetry, global flip\n";
      for(int a=0; a<L; a++) out << a << " 1 " << a << ' ' << a%M << '\n';
      out << "# cluster energies for unique baths, eps[k]\n";
      for(double E : levels) out << E*energy_unit << ' ';
      out << "\n# block N K Sz size, creation targets, energies, auxiliary spin labels\n";
      for(int i=0; i<int(blocks.size()); i++)
      {
        const auto& b = blocks[i];
        out << i+1 << ' ' << b.N << " 0 " << b.Sz << ' ' << b.states.size();
        for(int target : b.target) out << ' ' << target+1;
        for(double E : b.energy) out << ' ' << E*energy_unit;
        for(int j=0; j<int(b.states.size()); j++) out << " 0";
        out << '\n';
      }
      out << "# Matrix elements: source, target, source size, target size; source-major order\n";
      for(int i=0; i<int(blocks.size()); i++)
        for(int a=0; a<L; a++)
        {
          const auto& b = blocks[i];
          if(b.target[a] < 0) out << i+1 << " 0 0 0\n";
          else
          {
            out << i+1 << ' ' << b.target[a]+1 << ' ' << b.states.size() << ' ' << blocks[b.target[a]].states.size();
            for(int row=0; row<b.creation[a].rows(); row++)
              for(int col=0; col<b.creation[a].cols(); col++)
                out << ' ' << b.creation[a](row,col);
            out << '\n';
          }
        }
      out << "HB2\n# UCoulomb: i j k l U_ijkl, H_int = 1/2 sum U_ijkl c_i^+ c_j^+ c_k c_l\n";
      for(int i=0; i<L; i++)
        for(int j=0; j<L; j++)
          for(int k=0; k<L; k++)
            for(int l=0; l<L; l++)
              if(U[i][j][k][l] != 0.0)
                out << i << ' ' << j << ' ' << k << ' ' << l << ' ' << U[i][j][k][l]*energy_unit << '\n';
      out << "# number of operators needed\n0\n";
      if(!out) throw std::runtime_error("Failed to write Rutgers CIX");
    }
  }
}
