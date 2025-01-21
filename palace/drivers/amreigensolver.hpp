#ifndef PALACE_DRIVERS_AMR_EIGEN_SOLVER_HPP
#define PALACE_DRIVERS_AMR_EIGEN_SOLVER_HPP

#include <utility>
#include <vector>
#include <memory>
#include <tuple>

namespace palace
{

class IoData;
class Mesh;
class ErrorIndicator;

// This driver does repeated solves with local refinement near the junction
// Returns the final (ErrorIndicator, vsize) from the last solve
std::pair<ErrorIndicator, long long int>
AMREigenSolver(IoData &iodata,
               std::vector<std::unique_ptr<Mesh>> &mesh);

} // end namespace palace

#endif  // PALACE_DRIVERS_AMR_EIGEN_SOLVER_HPP
