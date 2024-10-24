/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */




#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/function_parser.h>

#include <fstream>
#include <iostream>
#include <math.h>




namespace Step26
{
  using namespace dealii;

  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation();
    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve_time_step();
    void output_results() const;
    
    

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;

    double       time;
    double       time_step;
    unsigned int timestep_number;
  };
 
  
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>()
      , period(0.2)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    const double period;
  };



  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());
    const double time = this->get_time();
    const double pi=M_PI;
    return std::pow(sin(pi*p[0])*cos(time),3)  -sin(pi*p[0])*sin(time)+(pi*pi  - 1)*sin(pi*p[0])*cos(time);
  }



  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };



  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & ,
                                    const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }

 
  template <int dim>
  HeatEquation<dim>::HeatEquation()
    : fe(1)
    , dof_handler(triangulation)
    , time_step(1. / 500)
  {}


  
  template <int dim>
  void HeatEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                            types::boundary_id(0),
                                            Functions::ZeroFunction<dim>(),
                                            constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                            types::boundary_id(1),
                                            Functions::ZeroFunction<dim>(),
                                            constraints);
  
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                     true);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }


  template <int dim>
  void HeatEquation<dim>::assemble_system()
  {
    system_matrix = 0;
    system_rhs = 0;
    Vector<double> tmp;
    Vector<double> forcing_terms;
    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());
    
    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
    std::vector<double> old_solution_values(n_q_points);

    RightHandSide<dim> rhs_function;
    rhs_function.set_time(time);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        
        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        fe_values.get_function_values(old_solution,old_solution_values);

        for(const unsigned int q_index : fe_values.quadrature_point_indices()){
          // u^{n-1}
          const double u_old = old_solution_values[q_index]; 
          for(const unsigned int i : fe_values.dof_indices()){
            for(const unsigned int j : fe_values.dof_indices()){
              
              cell_matrix(i, j) += (1.0 / time_step) * fe_values.shape_value(i, q_index) *
                                    fe_values.shape_value(j, q_index) *
                                    fe_values.JxW(q_index);

              
              cell_matrix(i, j) += 1*fe_values.shape_grad(i, q_index) *
                                    fe_values.shape_grad(j, q_index) *
                                    fe_values.JxW(q_index);

            }

            double rhs_value = rhs_function.value(fe_values.quadrature_point(q_index));
            // Mu^{n-1}
            cell_rhs(i) += (1.0 / time_step) * u_old *
                            fe_values.shape_value(i, q_index) * fe_values.JxW(q_index);
            // nonlinear part
            cell_rhs(i) += -u_old * (u_old * u_old - 1.0) *
                            fe_values.shape_value(i, q_index) * fe_values.JxW(q_index);
            // forcing term
            cell_rhs(i) += rhs_value *  fe_values.shape_value(i, q_index)* fe_values.JxW(q_index);
          }
        }

        cell->get_dof_indices(local_dof_indices);

    constraints.distribute_local_to_global(cell_matrix,cell_rhs,local_dof_indices,system_matrix,system_rhs);
        
        // for(unsigned int i : fe_values.dof_indices()){
        //   for(unsigned int j : fe_values.dof_indices()){
        //     system_matrix.add(local_dof_indices[i], 
        //                       local_dof_indices[j],
        //                       cell_matrix(i, j));
        //   }

        //   system_rhs(local_dof_indices[i]) += cell_rhs(i);
        // }
    }
    
    // VectorTools::create_right_hand_side(dof_handler,
    //                                     QGauss<dim>(fe.degree + 1),
    //                                     rhs_function,
    //                                     tmp);
    // forcing_terms = tmp;
    // forcing_terms *= time_step;

    // system_rhs += forcing_terms;
  }



    
  template <int dim>
  void HeatEquation<dim>::solve_time_step()
  {
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    std::cout << "     " << solver_control.last_step() << " CG iterations."
              << std::endl;
  }



  template <int dim>
  void HeatEquation<dim>::output_results() const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U");

    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
  }


          
  template <int dim>
  void HeatEquation<dim>::run()
  {
    const unsigned int initial_global_refinement       = 5;
    const std::string initial_condition = "sin(" + std::to_string(M_PI) + "*x)";

    GridGenerator::hyper_cube(triangulation, -1, 1, true);
    triangulation.refine_global(initial_global_refinement);

    setup_system();

    unsigned int pre_refinement_step = 0;

    // Vector<double> tmp;
    // Vector<double> forcing_terms;

  start_time_iteration:

    time            = 0.0;
    timestep_number = 0;

    // tmp.reinit(solution.size());
    // forcing_terms.reinit(solution.size());

    //initial condition: sin(pi*x)
    VectorTools::interpolate(dof_handler,
                              FunctionParser<dim>(initial_condition),
                              old_solution);
    solution = old_solution;

    output_results();

    while (time <= 0.5)
      {
        time += time_step;
        ++timestep_number;

        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;
        assemble_system();
        
        solve_time_step();

        output_results();

        old_solution = solution;
      }
  }
}




int main()
{
  try
    {
      using namespace Step26;

      HeatEquation<2> heat_equation_solver;
      heat_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
