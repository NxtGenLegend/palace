#include "amreigensolver.hpp"

#include <cmath>    // for abs, etc
#include <iostream>
#include "utils/communication.hpp"
#include "utils/iodata.hpp"
#include "fem/mesh.hpp"
#include "drivers/eigensolver.hpp"     // For EigenSolver
#include "models/spaceoperator.hpp"     // So we can create SpaceOperator
#include "fem/errorindicator.hpp"       // If needed for final indicators

// MFEM includes
#include <mfem.hpp>

namespace palace
{

std::pair<ErrorIndicator, long long int>
AMREigenSolver(IoData &iodata,
               std::vector<std::unique_ptr<Mesh>> &mesh)
{
  // We store old frequencies & old junction energies
  // We'll do a simple approach for freq: storing the fundamental freq only
  double old_freq = -1.0;
  double old_junc = -1.0;

  // Tolerances
  double freq_tol = iodata.solver.eigenmode.tol;
  double junc_tol = iodata.solver.eigenmode.junction_tol;
  int required_passes = iodata.solver.eigenmode.required_passes;

  int freq_consec = 0;
  int junc_consec = 0;

  const int max_adapt_its = 10;
  ErrorIndicator final_indicator;
  long long int final_vsize = 0;

  for (int iter = 1; iter <= max_adapt_its; iter++)
  {
    Mpi::Print("\n===== EIGEN-AMR Iteration #{} =====\n", iter);

    // (1) Solve the eigenproblem
    EigenSolver solver(iodata);
    auto [indicator, vsize, jenergy] = solver.Solve(mesh); 
    final_indicator = indicator;  // store 
    final_vsize = vsize;

    // (2) Compare fundamental freq with old_freq
    // For now, we never actually retrieved the fundamental freq from Solve()
    // unless you changed it to also return "fund_freq". We can do that easily:
    double new_freq = 0.0; // placeholder
    // E.g. you might store in Solve() -> e.g. "fund_freq" 
    // Then do auto [indicator, vsize, jenergy, fund_freq] = solver.Solve(...);

    double freq_diff = 0.0;
    if (old_freq > 0.0)
    {
      freq_diff = std::abs(new_freq - old_freq)/std::abs(old_freq);
    }
    bool freq_ok = (freq_diff < freq_tol);
    if (freq_ok) freq_consec++;
    else freq_consec = 0;
    old_freq = new_freq; // update

    // (3) Compare new_junc with old_junc
    double junc_diff = 0.0;
    if (old_junc > 0.0)
    {
      junc_diff = std::abs(jenergy - old_junc)/std::abs(old_junc);
    }
    bool junc_ok = (junc_diff < junc_tol);
    if (junc_ok) junc_consec++;
    else junc_consec = 0;
    old_junc = jenergy;

    Mpi::Print("   fundamental freq diff = {:.3e}, junc diff = {:.3e}\n",
               freq_diff, junc_diff);

    // check if both are converged enough times
    if (freq_consec >= required_passes && junc_consec >= required_passes)
    {
      Mpi::Print("Converged after {} iterations!\n", iter);
      return {final_indicator, final_vsize};
    }

    // (4) Local refinement near junction if not converged
    if (iter < max_adapt_its)
    {
      Mpi::Print("Local refinement near junction region...\n");
      // For each submesh in 'mesh'
      for (auto &msh : mesh)
      {
        auto &pm = msh->Get(); // returns a non-const mfem::ParMesh & 
        mfem::Array<int> elem_marker(pm.GetNE());
        elem_marker = 0;

        // Build a temporary SpaceOperator to get the junction elements
        SpaceOperator tmp_op(iodata, mesh);
        auto junc_elems = tmp_op.GetJunctionElements();
        for (int e : junc_elems)
        {
          elem_marker[e] = 1; 
        }

        // Now do local refinement
        pm.Refine(elem_marker);
        pm.ReorientTetMesh(); 
        // pm.Rebalance(); // If in parallel, might want to do it
        pm.Finalize();  // finalize
      }
    }
  }

  // If we exit here, we never got enough passes within max_adapt_its
  Mpi::Print("WARNING: Max outer iters reached, returning last solution.\n");
  return {final_indicator, final_vsize};
}

} // end namespace palace
