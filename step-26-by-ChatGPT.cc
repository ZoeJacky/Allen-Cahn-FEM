#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/function_lib.h>


#include <iostream>
#include <fstream>

using namespace dealii;

template <int dim>
class AllenCahn
{
public:
    AllenCahn();
    void run();

private:
    void setup_system();
    void assemble_system();
    void solve_time_step();
    void output_results(const unsigned int timestep) const;
    void time_loop();

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution_old;
    Vector<double> solution_new;
    Vector<double> system_rhs;

    double time_step;
    double total_time;
    unsigned int n_time_steps;
};

template <int dim>
AllenCahn<dim>::AllenCahn()
    : fe(1), dof_handler(triangulation), time_step(0.1), total_time(1.0)
{
    n_time_steps = total_time / time_step;
}

template <int dim>
void AllenCahn<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    solution_old.reinit(dof_handler.n_dofs());
    solution_new.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // Set initial condition (for example, a random perturbation)
    // VectorTools::interpolate(dof_handler, ConstantFunction<dim>(0.1), solution_old);
    VectorTools::interpolate(dof_handler, Functions::ConstantFunction<dim>(0.1), solution_old);

}

template <int dim>
void AllenCahn<dim>::assemble_system()
{
    system_matrix = 0;
    system_rhs = 0;

    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> old_solution_values(n_q_points);
    std::vector<Tensor<1, dim>> old_solution_gradients(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        local_matrix = 0;
        local_rhs = 0;

        fe_values.reinit(cell);

        fe_values.get_function_values(solution_old, old_solution_values);
        fe_values.get_function_gradients(solution_old, old_solution_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double old_value = old_solution_values[q];
            const Tensor<1, dim> old_gradient = old_solution_gradients[q];

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values.shape_value(i, q);
                const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double phi_j = fe_values.shape_value(j, q);
                    const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);

                    // Time-stepping mass matrix + Laplace term
                    local_matrix(i, j) += (phi_i * phi_j / time_step + grad_phi_i * grad_phi_j) * fe_values.JxW(q);
                }

                // Nonlinear term (u^{n-1} * (u^{n-1})^2 - 1)
                local_rhs(i) += (old_value * (old_value * old_value - 1) * phi_i) * fe_values.JxW(q);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        system_matrix.add(local_dof_indices, local_matrix);
        system_rhs.add(local_dof_indices, local_rhs);
    }

    // Add old solution contributions
    // system_rhs.add(solution_old, 1.0 / time_step);
    system_rhs.add(1.0 / time_step, solution_old);

}

template <int dim>
void AllenCahn<dim>::solve_time_step()
{
    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult(solution_new, system_rhs);
}

template <int dim>
void AllenCahn<dim>::output_results(const unsigned int timestep) const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_new, "solution");
    data_out.build_patches();

    std::ofstream output("solution-" + std::to_string(timestep) + ".vtk");
    data_out.write_vtk(output);
}

template <int dim>
void AllenCahn<dim>::time_loop()
{
    for (unsigned int timestep = 0; timestep < n_time_steps; ++timestep)
    {
        std::cout << "Time step " << timestep << "/" << n_time_steps << std::endl;

        assemble_system();
        solve_time_step();

        output_results(timestep);

        // Update old solution
        solution_old = solution_new;
    }
}

template <int dim>
void AllenCahn<dim>::run()
{
    std::cout << "Solving Allen-Cahn equation..." << std::endl;

    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(5);  // Refining the grid

    setup_system();
    time_loop();
}

int main()
{
    try
    {
        AllenCahn<2> allen_cahn_problem;
        allen_cahn_problem.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl << "Exception: " << exc.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl << "Unknown exception!" << std::endl;
        return 1;
    }

    return 0;
}
