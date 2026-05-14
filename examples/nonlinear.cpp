#include <iostream>

#include "mfem.hpp"
#include <fstream>
#include <iostream>

// refuse to use namespace std or namespace mfem
using namespace mfem;

class NLDiffusionIntegrator : public NonlinearFormIntegrator
{
public:
  NLDiffusionIntegrator(Coefficient & k,
                        Coefficient & dk_du,
                        const GridFunction * gf,
                        const IntegrationRule * ir = nullptr)

  : _diffusion_integ(k, ir),
  _grad_trial(gf),
  _neg_grad_trial(_neg_one, _grad_trial),
  _neg_dk_du_grad_trial(dk_du, _neg_grad_trial),
  _weak_div_integ(_neg_grad_trial)
  {
    _sum.AddIntegrator(&_diffusion_integ);
    _sum.AddIntegrator(&_weak_div_integ);
  }

  virtual void AssembleElementVector(const FiniteElement & el,
                                     ElementTransformation & Tr,
                                     const Vector & elfun,
                                     Vector & elvect) {
    _diffusion_integ.AssembleElementVector(el, Tr, elfun, elvect);
  }

  virtual void AssembleElementGrad(const FiniteElement & el,
                                   ElementTransformation & Tr,
                                   const Vector & elfun,
                                   DenseMatrix & elmat) {
    _sum.AssembleElementGrad(el, Tr, elfun, elmat);
  }

protected:
  DiffusionIntegrator _diffusion_integ;
  ConstantCoefficient _neg_one{-1.0};
  GradientGridFunctionCoefficient _grad_trial;
  ScalarVectorProductCoefficient _neg_grad_trial;
  ScalarVectorProductCoefficient _neg_dk_du_grad_trial;
  MixedScalarWeakDivergenceIntegrator _weak_div_integ;
  SumIntegrator _sum{0};
};

// Trivial overload of NonlinearForm that makes sure we update the
// Gridfunction U every iteration
class NLWithUpdate : public ParNonlinearForm {
public:
  NLWithUpdate(ParFiniteElementSpace* f, ParGridFunction& u) : ParNonlinearForm(f), _u(u), _blf(f) {}
  NLWithUpdate() = delete;

  // basically copy everything from EquationSystem::Mult
  void Mult(const Vector & sol, Vector & residual) const override {
    const Vector temp_sol( const_cast< Vector& >(sol) ); temp_sol.SyncAliasMemory(temp_sol);
    // set the gridfunction from this vector (emulate SetTrialVariablesFromTrueVectors)
    // i don't think we need this intermediate vector. I think EquationSystem does
    // this because of the block nature of the solution vectors
    _u.Distribute(&temp_sol);

    // finished with SetTrialVariablesFromTrueVectors, back to Mult
    Vector temp_res(residual);
    ParNonlinearForm::Mult(sol, residual);
    temp_res.SyncAliasMemory(temp_res);

    // finally, call AddMult
    // "dereferencing" this object just fetches oper, 
    // which is the HypreParMatrix hidden within
    _linear_operator->AddMult(sol, residual);

    sol.HostRead();
    residual.HostRead();
  }

  Operator& GetGradient(const Vector& u) const override {
    const Vector update_vector(const_cast<Vector &>(u));
    mfem::HypreParMatrix * nlf_jac =
      dynamic_cast<mfem::HypreParMatrix *>(&ParNonlinearForm::GetGradient(update_vector));

    // in the original, we did ParAdd on _h_blocks(i, i) with nlf_jac.
    // so here, the first argument will be aux_a
    HypreParMatrix* _temp_jacobian = ParAdd(_aux_a, nlf_jac);

    // again, this is a consequence of the block nature of EquationSystem
    _jacobian.Reset(_temp_jacobian);

    return *_jacobian;
  }

  // in the nl diffusion example we set the rhs values through a call to FormLinearSystem
  // which we replicate here to set the RHS values correctly and then discard the intermediate
  // stuff...
  void FormLinearSystem(ParGridFunction& u, Vector& x, ParLinearForm& b, Vector& rhs, Array<int>& ess_tdof_list) {
    _aux_a = new mfem::HypreParMatrix;

    _blf.SetAssemblyLevel(AssemblyLevel::LEGACY);
    _blf.Assemble();
    // dummy_blf.Finalize();
    _blf.FormLinearSystem(ess_tdof_list, u, b, *_aux_a, x, rhs);

    // we don't delete aux_a in the destructor. Hope it doesn't leak
    _linear_operator.Reset(_aux_a);
  }

protected:
  ParGridFunction& _u;
  mutable mfem::OperatorHandle _linear_operator;
  mutable mfem::OperatorHandle _jacobian;
  mfem::HypreParMatrix * _aux_a;
  ParBilinearForm _blf;
};

/*
BOUNDARY CONDITIONS
  - top side needs to be held to 3.0 (side set number 3)
  - bottom side to 1.0 (side set number 1)
  -
  - we need to find the boundary markers that correspond correctly here...

Initial Conditions:
  - we need to call GridFunction::ProjectCoefficient with the
    coefficient we create from the functor. That makes sure it gets in
*/


// supposed to be 2*y+1
// use this to initialise the functioncoefficient
real_t initial(const Vector& vec) { return 2*vec[1]+1; }

int main() {
  Mpi::Init();
  Hypre::Init();
  
  // mfem function 
  const char *mesh_file = "../data/square.e";
  int order = 1;

  // read the mesh in
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();
  ParMesh pmesh(MPI_COMM_WORLD, mesh);

  // Fe collection - H1 elements, first order
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fespace(&pmesh, &fec);

  // solution vector
  ParGridFunction u(&fespace);

  // initial conditions
  FunctionCoefficient intial_conds(initial);
  u.ProjectCoefficient(intial_conds);

  // boundary conditions
  // THIS WILL ONLY WORK FOR THE SQUARE MESH!!
  Array<int> ess_bdr(pmesh.bdr_attributes.Max()); ess_bdr = 0;
  Array<int> ess_tdof_list;

  // bottom boundary - this is sideset 1 in cubit
  ess_bdr[0] = 1;
  // top boundary - this is sideset 3 in cubit
  ess_bdr[2] = 1;
  fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // bottom bc - hold it to 1.0. this is sideset 1 in cubit
  ConstantCoefficient one(1.0);
  // Array<int> btm_bdr(1);
  ess_bdr = 0; ess_bdr[0] = 1;
  u.ProjectBdrCoefficient(one, ess_bdr);

  // top BC - we want to hold it to 3.0. this is sideset 3 in cubit
  ConstantCoefficient three(3.0);
  ess_bdr = 0; ess_bdr[2] = 1;
  u.ProjectBdrCoefficient(three, ess_bdr);

  // Linear form - commented out since inactive in nldiffusion.i
  ParLinearForm b(&fespace); b = 0.0;
  Vector rhs;
  // b.AddDomainIntegrator(new DomainLFIntegrator(one));
  b.Assemble();

  ConstantCoefficient dk_du_coef(1.0);
  GridFunctionCoefficient k_coef(&u);

  // Set up nl form
  NLWithUpdate a(&fespace, u);
  // NonlinearForm a(&fespace);
  a.AddDomainIntegrator(new NLDiffusionIntegrator(k_coef, dk_du_coef, &u));

  a.SetEssentialTrueDofs(ess_tdof_list);

  Vector X;

  a.FormLinearSystem(u, X, b, rhs, ess_tdof_list);

  // SOLVE
  NewtonSolver newton;
  newton.SetOperator(a);
  newton.SetRelTol(1e-8);
  newton.SetAbsTol(1e-8);
  newton.SetMaxIter(20);
  newton.SetPrintLevel(1);

  GMRESSolver linear_solver;
  linear_solver.SetAbsTol(1e-10);
  linear_solver.SetMaxIter(10);
  linear_solver.SetPrintLevel(1);
  
  newton.SetSolver(linear_solver);

  u.GetTrueDofs(X);
  newton.Mult(rhs,X);

  // write solution back - should be RecoverFEMSolution
  u.SetFromTrueDofs(X);

  // can we vis ?
  std::ofstream mesh_ofs("nl.mesh");
  mesh_ofs.precision(8);
  pmesh.Print(mesh_ofs);

  std::ofstream u_ofs("u.gf");
  u_ofs.precision(8);
  u.Save(u_ofs);

  bool visualization = true;
  // 14. Send the solution by socket to a GLVis server.
  if (visualization)
  {
    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n" << mesh << u << std::flush;
  }

  return 0;
}
