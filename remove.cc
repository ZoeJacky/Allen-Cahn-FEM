template <int dim>
void HeatEquation<dim>::assemble_system()
{
  system_matrix = 0;
  system_rhs = 0;

  QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Loop over all cells in the triangulation
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;

    std::vector<double> old_solution_values(n_q_points);

    // Get old solution values at the quadrature points
    fe_values.get_function_values(old_solution, old_solution_values);

    for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
    {
      const double u_old = old_solution_values[q_index]; // u^{n-1}
      const double nonlinear_term = u_old * (u_old * u_old - 1.0); // u^{n-1}((u^{n-1})^2 - 1)

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double phi_i = fe_values.shape_value(i, q_index);
        const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q_index);

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const double phi_j = fe_values.shape_value(j, q_index);
          const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q_index);

          // Assemble system matrix (time-stepping and diffusion)
          cell_matrix(i, j) += (phi_i * phi_j / time_step) * fe_values.JxW(q_index); // Mass term (1/dt)
          cell_matrix(i, j) += (grad_phi_i * grad_phi_j) * fe_values.JxW(q_index);   // Diffusion term (Laplacian)
        }

        // Assemble RHS (old solution and nonlinear term)
        cell_rhs(i) += (phi_i * old_solution_values[q_index] / time_step) * fe_values.JxW(q_index); // old solution term
        cell_rhs(i) += phi_i * nonlinear_term * fe_values.JxW(q_index); // nonlinear term
      }
    }

    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));

      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  }
}
