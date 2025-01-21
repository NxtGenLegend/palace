// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "errorindicator.hpp"

#include <mfem/general/forall.hpp>
// CUSTOM CONVERGENCE
#include <models/spaceoperator.hpp>

namespace palace
{

void ErrorIndicator::AddIndicator(const Vector &indicator)
{
  if (n == 0)
  {
    local = indicator;
    n = 1;
    return;
  }

  // The average local indicator is used rather than the indicator for the maximum
  // error to drive the adaptation, to account for a local error that might be marginally
  // important to many solves, rather than only large in one solve.
  MFEM_ASSERT(local.Size() == indicator.Size(),
              "Unexpected size mismatch for ErrorIndicator::AddIndicator!");

  // The local indicators must be squared before combining, so that the global error
  // calculation is valid:
  //                            E = √(1/N ∑ₙ ∑ₖ ηₖₙ²)
  // from which it follows that:
  //                            E² = 1/N ∑ₙ ∑ₖ ηₖₙ²
  //                               = 1/N ∑ₙ Eₙ²
  // Namely the average of the global error indicators included in the reduction.
  // Squaring both sides means the summation can be rearranged, and then the local error
  // indicators become:
  //                            eₖ = √(1/N ∑ₙ ηₖₙ²)
  const bool use_dev = local.UseDevice() || indicator.UseDevice();
  const int N = local.Size();
  const int Dn = n;
  const auto *DI = indicator.Read();
  auto *DL = local.ReadWrite();
  mfem::forall_switch(
      use_dev, N, [=] MFEM_HOST_DEVICE(int i)
      { DL[i] = std::sqrt((DL[i] * DL[i] * Dn + DI[i] * DI[i]) / (Dn + 1)); });

  // More samples have been added, update for the running average.
  n += 1;
}

// CUSTOM CONVERGENCE
bool JunctionConvergenceMonitor::AddMeasurement(
    const Vector &field_mag, const SpaceOperator &space_op)
{
    // 1) Identify the junction elements from the space operator
    std::vector<int> junction_elems = space_op.GetJunctionElements();

    if (!reported_junction_count && !junction_elems.empty())
    {
        Mpi::Print(" Number of junction elements: {}\n", junction_elems.size());
        reported_junction_count = true;
    }

    // 2) Compute "junction energy" by summing |E|^2 * volume of each junction element
    //    Use mesh.GetElementVolume(elem), which is *const-friendly*.
    const auto &mfem_mesh = space_op.GetMesh(); // returns a const reference
    double current_energy = 0.0;
    for (int elem : junction_elems)
    {
        double value = field_mag[elem]; // The field magnitude in that element
        double volume = mfem_mesh.GetElementVolume(elem);
        current_energy += (value * value) * volume;
    }

    // 3) First iteration, no previous energy to compare
    if (prev_energy < 0.0)
    {
        prev_energy = current_energy;
        return false; // not converged yet
    }

    // 4) Relative change in junction energy
    double change = std::abs((current_energy - prev_energy) / prev_energy);

    Mpi::Print(" Junction energy change: %.3e%% (need < %.3e%%, %d consecutive)\n",
               change * 100.0, tol * 100.0, consecutive_passes);

    if (change < tol)
    {
        consecutive_passes++;
    }
    else
    {
        consecutive_passes = 0;
    }

    prev_energy = current_energy;
    return (consecutive_passes >= required_passes);
}

}  // namespace palace
