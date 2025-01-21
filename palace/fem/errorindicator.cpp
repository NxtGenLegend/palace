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
    double current_energy = 0.0;
    auto junction_elements = space_op.GetJunctionElements();

    if (!reported_junction_count && !junction_elements.empty()) {
        Mpi::Print(" Number of junction elements: {}\n", junction_elements.size());
        reported_junction_count = true;
    }

    const auto& mfem_mesh = space_op.GetMesh().Get();
    for(int elem : junction_elements) {
        const double value = field_mag[elem];
        current_energy += value * value * mfem_mesh.GetElementVolume(elem);
    }

    if (prev_energy < 0) {
        prev_energy = current_energy;
        return false;
    }

    double change = std::abs((current_energy - prev_energy) / prev_energy);
    
    Mpi::Print(" Junction energy change: {:.3e}% (needed < {:.3e}%, {} consecutive)\n",
               change * 100.0, tol * 100.0, consecutive_passes);
    
    if (change < tol) {
        consecutive_passes++;
    } else {
        consecutive_passes = 0;
    }

    prev_energy = current_energy;
    return consecutive_passes >= required_passes;
}

}  // namespace palace
