#ifndef PALACE_DRIVERS_AMR_EIGEN_SOLVER_HPP
#define PALACE_DRIVERS_AMR_EIGEN_SOLVER_HPP

#include <utility>
#include <vector>
#include <memory>
#include <tuple>
#include "drivers/basesolver.hpp"

namespace palace
{

class IoData;
class Mesh;
class ErrorIndicator;

class AMREigenSolver : public BaseSolver
{
private:
  IoData &iodata;
  bool root;
  std::size_t size;
  int num_thread;
  const char *git_tag;

public:
  AMREigenSolver(IoData &iodata, bool root, std::size_t size, int num_thread,
                 const std::string &git_tag)
      : iodata(iodata), root(root), size(size), num_thread(num_thread), git_tag(git_tag) {}
// This driver does repeated solves with local refinement near the junction
// Returns the final (ErrorIndicator, vsize) from the last solve
  std::pair<ErrorIndicator, long long int>
    Solve(const std::vector<std::unique_ptr<Mesh>> &mesh) const override;
  using BaseSolver::BaseSolver;
};

} // end namespace palace

#endif  // PALACE_DRIVERS_AMR_EIGEN_SOLVER_HPP
