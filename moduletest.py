import sympy as sp
import ipywidgets as widgets
from IPython.display import display, Latex
import os
import re
import subprocess
import requests
from google.colab import files


class CircuitAnalyzer:
    """General electronic RC circuit analyzer"""

    def __init__(self, num_nodes, edges_R, edges_C, voltage_source_map, loops,
                 q_branches_map_predefined=None, circuit_diagram_url=None):
        self.num_nodes = num_nodes
        self.edges_R = edges_R
        self.edges_C = edges_C
        self.voltage_source_map = voltage_source_map
        self.loops = loops
        self.q_branches_map_predefined = q_branches_map_predefined
        self.circuit_diagram_url = circuit_diagram_url

        if self.circuit_diagram_url:
            self._download_circuit_diagram()

        self._setup_branches()
        self._setup_parameters()

    # ─────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────

    def _download_circuit_diagram(self):
        """Download circuit diagram from provided URL"""
        try:
            response = requests.get(self.circuit_diagram_url)
            response.raise_for_status()
            content = response.text
            content = content.replace('\\nnode', '\\node')
            content = content.replace('\\nodraw', '\\node')
            content = re.sub(r'\\n([a-zA-Z])', r'\\\1', content)
            content = re.sub(r'\\ode\b', r'\\node', content)
            with open('circuit_diagram.tex', 'w', encoding='utf-8') as f:
                f.write(content)
            print("Circuit diagram downloaded and processed successfully")
            return True
        except Exception as e:
            print(f"❌ Error downloading diagram: {e}")
            print("⚠️ Using default template")
            self._create_default_diagram()
            return False

    def _create_default_diagram(self):
        default_content = r"""
\documentclass[a4paper,12pt]{article}
\usepackage{amsmath,geometry,circuitikz}
\usepackage[utf8]{inputenc}
\geometry{margin=1in}
\begin{document}
\begin{figure}[h!]
\centering
\begin{circuitikz}[scale=1.2]
\node[draw, text width=10cm, align=center, fill=blue!10] at (0,0) {
    Circuit diagram not available. \\
    Please provide a valid circuit diagram URL or TeX file. \\
    The equations will still be generated correctly.
};
\end{circuitikz}
\caption{Generic Circuit Analysis Template}
\end{figure}
\end{document}
"""
        with open('circuit_diagram.tex', 'w', encoding='utf-8') as f:
            f.write(default_content)

    def _setup_branches(self):
        all_R_branches = [tuple(sorted(edge[:2])) for edge in self.edges_R]
        all_C_branches = [tuple(sorted(edge[:2])) for edge in self.edges_C]
        self.all_branches = sorted(list(set(all_R_branches + all_C_branches)),
                                   key=lambda x: (x[0], x[1]))
        self.capacitor_branches = set(all_C_branches)
        self.resistor_branches = set(all_R_branches)
        print(f"All branches: {self.all_branches}")
        print(f"Capacitor branches: {self.capacitor_branches}")
        print(f"Resistor branches: {self.resistor_branches}")

    def _setup_parameters(self):
        self.N = len(self.all_branches)
        self.num_independent = self.N - (self.num_nodes - 1)
        self.num_capacitors = len(self.capacitor_branches)
        if self.num_capacitors > self.num_independent:
            raise ValueError(
                f"Too many capacitors! Need at most {self.num_independent} but have {self.num_capacitors}"
            )
        self.num_additional = self.num_independent - self.num_capacitors
        # Only show branches that yield a solvable KCL system
        candidate_branches = [b for b in self.all_branches if b not in self.capacitor_branches]
        self.branches_for_additional = self._filter_valid_branches(candidate_branches)
        print(f"Total branches (N): {self.N}")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Independent currents needed: {self.num_independent}")
        print(f"Capacitor branches: {self.num_capacitors}")
        print(f"Additional independent branches to select: {self.num_additional}")
        print(f"Valid selectable branches: {self.branches_for_additional}")

    def _filter_valid_branches(self, candidate_branches):
        """Return only branches that yield a solvable KCL system.
        A branch is valid if adding it as an independent current keeps
        the KCL incidence matrix full rank over the dependent unknowns.
        """
        t = sp.Symbol('t')
        letters = ['x','y','z','u','v','w','p','q','r','s','m','n','k','j','h']

        # Build cap-only q_map and orientations
        cap_q_map = {}
        cap_orientations = {}
        if self.q_branches_map_predefined:
            for q_name, branch in self.q_branches_map_predefined.items():
                bs = tuple(sorted(branch))
                if bs in self.capacitor_branches:
                    cap_q_map[bs] = q_name
                    cap_orientations[bs] = branch
        else:
            for i, branch in enumerate(sorted(self.capacitor_branches)):
                q_name = f'q_{i+1}'
                cap_q_map[branch] = q_name
                cap_orientations[branch] = branch

        valid = []
        for test_branch in candidate_branches:
            q_map_test = dict(cap_q_map)
            orient_test = dict(cap_orientations)
            bs = tuple(sorted(test_branch))
            q_map_test[bs] = f'q_{len(cap_q_map)+1}'
            orient_test[bs] = test_branch

            for branch in self.all_branches:
                if branch not in orient_test:
                    orient_test[branch] = branch

            all_q_names = sorted(q_map_test.values(), key=lambda x: int(x.split('_')[1]))
            q_to_l = {q: letters[i] for i, q in enumerate(all_q_names)}
            q_funcs_t  = {q: sp.Function(l)(t) for q, l in q_to_l.items()}
            dq_funcs_t = {q: sp.diff(f, t) for q, f in q_funcs_t.items()}

            branch_currents_t = {}
            dep_syms_t = []
            for branch in self.all_branches:
                if branch in q_map_test:
                    branch_currents_t[branch] = dq_funcs_t[q_map_test[branch]]
                else:
                    s = sp.Symbol(f'i_{branch[0]}_{branch[1]}')
                    branch_currents_t[branch] = s
                    dep_syms_t.append(s)

            if not dep_syms_t:
                valid.append(test_branch)
                continue

            kcl_rows = []
            for node in range(1, self.num_nodes + 1):
                row = sp.sympify(0)
                for branch in self.all_branches:
                    if node in branch:
                        ou, ov = orient_test[branch]
                        cur = branch_currents_t[branch]
                        if node == ou: row -= cur
                        else: row += cur
                kcl_rows.append(row)

            M = sp.Matrix([[row.coeff(s) for s in dep_syms_t] for row in kcl_rows])
            if M.rank() >= len(dep_syms_t):
                valid.append(test_branch)

        return valid

    # ─────────────────────────────────────────────
    # Widgets
    # ─────────────────────────────────────────────

    def create_widgets(self):
        self.dropdowns = []
        if self.num_additional > 0:
            for i in range(self.num_additional):
                default_index = min(i, len(self.branches_for_additional) - 1)
                dropdown = widgets.Dropdown(
                    options=[f"({b[0]}, {b[1]})" for b in self.branches_for_additional],
                    description=f'Branch for i_{self.num_capacitors + i + 1}:',
                    value=f"({self.branches_for_additional[default_index][0]}, "
                          f"{self.branches_for_additional[default_index][1]})",
                    style={'description_width': 'initial'}
                )
                self.dropdowns.append(dropdown)

        self.calculate_button = widgets.Button(
            description='Generate Equations',
            button_style='success',
            tooltip='Click to generate differential equations'
        )
        self.output = widgets.Output()
        self.calculate_button.on_click(self._on_calculate_clicked)

    def _validate_selections(self, selected_branches):
        errors = []
        seen = set()
        for branch in selected_branches:
            branch_sorted = tuple(sorted(branch))
            if branch_sorted in seen:
                errors.append(f"Duplicate branch selected: {branch}")
            seen.add(branch_sorted)
        for branch in selected_branches:
            if tuple(sorted(branch)) in self.capacitor_branches:
                errors.append(f"Cannot select capacitor branch: {branch}")
        for branch in selected_branches:
            if tuple(sorted(branch)) not in self.all_branches:
                errors.append(f"Branch {branch} does not exist in circuit")
        return errors

    def _on_calculate_clicked(self, b):
        with self.output:
            self.output.clear_output()
            try:
                selected_additional_branches = []
                for dropdown in self.dropdowns:
                    branch = tuple(map(int, re.findall(r'\d+', dropdown.value)))
                    selected_additional_branches.append(branch)
                errors = self._validate_selections(selected_additional_branches)
                if errors:
                    for error in errors:
                        print(f"❌ Error: {error}")
                    return
                self._perform_analysis(selected_additional_branches)
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                import traceback
                traceback.print_exc()

    # ─────────────────────────────────────────────
    # Capacitor Orientation
    # ─────────────────────────────────────────────

    def _auto_assign_capacitor_orientations(self):
        q_branches_map = {}
        if self.q_branches_map_predefined:
            predefined_branches = set()
            for q_name, branch_uv in self.q_branches_map_predefined.items():
                branch_sorted = tuple(sorted(branch_uv))
                if branch_sorted not in self.capacitor_branches:
                    print(f"⚠️ Warning: Predefined branch {branch_uv} for {q_name} is not a capacitor branch!")
                    continue
                if branch_sorted in predefined_branches:
                    print(f"⚠️ Warning: Duplicate branch {branch_uv} in predefined orientations!")
                    continue
                predefined_branches.add(branch_sorted)
                q_branches_map[q_name] = branch_uv
            q_counter = len(q_branches_map) + 1
            for branch in sorted(self.capacitor_branches):
                if branch not in predefined_branches:
                    q_name = f'q_{q_counter}'
                    q_branches_map[q_name] = branch
                    q_counter += 1
                    print(f"ℹ️ Auto-assigned: {q_name} → {branch}")
        else:
            q_counter = 1
            for branch in sorted(self.capacitor_branches):
                q_name = f'q_{q_counter}'
                q_branches_map[q_name] = branch
                q_counter += 1
                print(f"ℹ️ Auto-assigned: {q_name} → {branch}")
        print(f"Final capacitor orientations: {q_branches_map}")
        return q_branches_map

    # ─────────────────────────────────────────────
    # Main Analysis
    # ─────────────────────────────────────────────

    def _perform_analysis(self, selected_additional_branches):
        branch_orientations = {}
        q_branches_map_predefined = self._auto_assign_capacitor_orientations()

        q_map = {}
        q_orientation_map = {}

        for q_name, branch in q_branches_map_predefined.items():
            branch_sorted = tuple(sorted(branch))
            q_map[branch_sorted] = q_name
            q_orientation_map[q_name] = branch
            branch_orientations[branch_sorted] = branch

        for i, branch_uv in enumerate(selected_additional_branches):
            q_name = f'q_{self.num_capacitors + i + 1}'
            branch_sorted = tuple(sorted(branch_uv))
            q_map[branch_sorted] = q_name
            q_orientation_map[q_name] = branch_uv
            branch_orientations[branch_sorted] = branch_uv

        for branch in self.all_branches:
            if branch not in branch_orientations:
                branch_orientations[branch] = branch

        t = sp.symbols('t')
        R, C, a = sp.symbols('R C a')

        latin_letters = ['x', 'y', 'z', 'u', 'v', 'w', 'p', 'q', 'r', 's',
                         'm', 'n', 'k', 'j', 'h', 'g', 'f', 'd', 'b', 'l']
        num_q = len(q_map)
        if num_q > len(latin_letters):
            for idx in range(num_q - len(latin_letters)):
                latin_letters.append(f'x_{len(latin_letters) + idx + 1}')

        chosen_letters = latin_letters[:num_q]
        q_symbols = sp.symbols(' '.join(chosen_letters))
        if isinstance(q_symbols, sp.Symbol):
            q_symbols = [q_symbols]

        q_to_letter = {}
        q_names_sorted = sorted(q_map.values(), key=lambda x: int(x.split('_')[1]))
        for i, q_name in enumerate(q_names_sorted):
            q_to_letter[q_name] = chosen_letters[i]

        q_funcs = {}
        dq_funcs = {}
        for q_name in q_names_sorted:
            letter = q_to_letter[q_name]
            q_funcs[q_name] = sp.Function(letter)(t)
            dq_funcs[q_name] = sp.diff(q_funcs[q_name], t)

        dq_subs = {
            dq_func: sp.Symbol(f'\\dot{{{q_to_letter[q_name]}}}')
            for q_name, dq_func in dq_funcs.items()
        }

        branch_currents = {}
        dependent_current_symbols = []

        for branch in self.all_branches:
            if branch in q_map:
                branch_currents[branch] = dq_funcs[q_map[branch]]
            else:
                dep_sym = sp.Symbol(f'i_{branch[0]}_{branch[1]}')
                branch_currents[branch] = dep_sym
                dependent_current_symbols.append(dep_sym)

        # KCL — use only n-1 independent equations (exclude last node)
        kcl_equations = []
        for node in range(1, self.num_nodes):
            current_sum = sp.sympify(0)
            for branch in self.all_branches:
                if node in branch:
                    orient_u, orient_v = branch_orientations[branch]
                    current = branch_currents[branch]
                    if node == orient_u:
                        current_sum -= current
                    else:
                        current_sum += current
            kcl_equations.append(current_sum)

        if dependent_current_symbols:
            try:
                if len(kcl_equations) < len(dependent_current_symbols):
                    print(f"⚠️ Under-determined system: {len(kcl_equations)} equations "
                          f"for {len(dependent_current_symbols)} unknowns.")
                    print("⚠️ Try selecting different branches for independent currents.")
                    return
                solutions = sp.solve(kcl_equations, dependent_current_symbols, dict=True)
                if not solutions:
                    print("❌ KCL system could not be solved.")
                    print("💡 Suggestion: Try selecting different branches.")
                    return
                solution_dict = solutions[0]
                for branch, current_expr in branch_currents.items():
                    if current_expr in solution_dict:
                        branch_currents[branch] = solution_dict[current_expr].simplify()
            except Exception as e:
                print(f"❌ Error while solving KCL: {e}")
                print("💡 Suggestion: Try selecting different branches.")
                return

        # KVL
        kvl_equations = []
        for loop in self.loops:
            if len(loop) < 3:
                continue
            loop_eq = sp.sympify(0)
            for j in range(len(loop) - 1):
                u, v = loop[j], loop[j + 1]
                branch_sorted = tuple(sorted((u, v)))
                if branch_sorted not in self.all_branches:
                    continue
                orient_u, orient_v = branch_orientations[branch_sorted]
                traversal_dir = 1 if (u, v) == (orient_u, orient_v) else -1
                current = branch_currents[branch_sorted]
                if branch_sorted in self.resistor_branches:
                    loop_eq -= traversal_dir * R * current
                if branch_sorted in self.capacitor_branches:
                    q_name = q_map[branch_sorted]
                    q_val = q_funcs[q_name]
                    loop_eq -= traversal_dir * (q_val / C)
                source_val = 0
                if (u, v) in self.voltage_source_map:
                    source_val = self.voltage_source_map[(u, v)]
                elif (v, u) in self.voltage_source_map:
                    source_val = -self.voltage_source_map[(v, u)]
                loop_eq += source_val
            final_eq = (loop_eq * C).subs(R, a / C).simplify()
            kvl_equations.append(sp.Eq(final_eq, 0))

        # Rewrite in terms of dot{x}, dot{y}, ...
        kvl_equations_new_vars = []
        for eq in kvl_equations:
            eq_sub = eq.lhs.subs(dq_subs)
            for q_name, letter in q_to_letter.items():
                eq_sub = eq_sub.subs(q_funcs[q_name], sp.Symbol(letter))
            kvl_equations_new_vars.append(sp.Eq(eq_sub, 0))

        # Solve each equation for one derivative
        solved_expressions = []
        used_vars = set()
        dq_symbols = [sp.Symbol(f'\\dot{{{letter}}}') for letter in q_to_letter.values()]

        for eq in kvl_equations_new_vars:
            lhs = eq.lhs
            present_vars = [var for var in dq_symbols if var in lhs.free_symbols]
            if not present_vars:
                solved_expressions.append(None)
                continue
            target_var = None
            for var in present_vars:
                if var not in used_vars:
                    target_var = var
                    used_vars.add(var)
                    break
            if target_var is None:
                target_var = present_vars[0]
            try:
                solutions = sp.solve(lhs, target_var)
                if solutions:
                    solved_expressions.append(sp.Eq(target_var, solutions[0].simplify()))
                else:
                    solved_expressions.append(None)
            except Exception:
                solved_expressions.append(None)

        # State-Space extraction — pass original equations with dq_funcs
        self._extract_state_space(q_to_letter, kvl_equations, q_funcs, dq_funcs)

        self._display_results(
            q_orientation_map, branch_currents, q_map, kvl_equations,
            kvl_equations_new_vars, solved_expressions, q_funcs, dq_funcs,
            q_to_letter, q_symbols, dq_subs, branch_orientations
        )

    # ─────────────────────────────────────────────
    # State-Space
    # ─────────────────────────────────────────────

    def _extract_state_space(self, q_to_letter, kvl_equations, q_funcs, dq_funcs):
        """Extract A and B matrices for dq/dt = Aq + Bu.

        Each KVL equation may contain multiple derivatives, so we must solve
        ALL equations simultaneously for ALL derivatives at once.
        """
        # ── voltage source symbols (positive, unique) ──────────────────────
        voltage_symbols = []
        seen_vsyms = set()
        for val in self.voltage_source_map.values():
            for atom in val.free_symbols:
                if atom not in seen_vsyms:
                    voltage_symbols.append(atom)
                    seen_vsyms.add(atom)

        # ── ordered lists ──────────────────────────────────────────────────
        q_names_sorted = sorted(q_to_letter.keys(), key=lambda x: int(x.split('_')[1]))
        dq_list   = [dq_funcs[q] for q in q_names_sorted]
        q_list    = [q_funcs[q]  for q in q_names_sorted]
        state_sym = [sp.Symbol(q_to_letter[q]) for q in q_names_sorted]

        n = len(dq_list)
        m = len(voltage_symbols)

        A_matrix = sp.zeros(n, n)
        B_matrix = sp.zeros(n, m)

        # ── solve all KVL equations simultaneously for all derivatives ─────
        lhs_list = [eq.lhs for eq in kvl_equations]
        try:
            all_sols = sp.solve(lhs_list, dq_list, dict=True)
        except Exception as e:
            print(f"⚠️ Could not solve for state-space: {e}")
            all_sols = []

        if not all_sols:
            print("⚠️ State-space extraction failed: no solution found.")
            self.A_matrix = A_matrix
            self.B_matrix = B_matrix
            self.state_vars = state_sym
            self.voltage_symbols = voltage_symbols
            return

        sol_dict = all_sols[0]

        for i, dq in enumerate(dq_list):
            if dq not in sol_dict:
                print(f"⚠️ No solution found for derivative {i+1}")
                continue
            rhs = sp.expand(sol_dict[dq])

            # substitute q(t) → plain symbol for coeff extraction
            for q_func, s_sym in zip(q_list, state_sym):
                rhs = rhs.subs(q_func, s_sym)

            # A row: coefficients of state variables
            for j, sv in enumerate(state_sym):
                A_matrix[i, j] = sp.simplify(rhs.coeff(sv))

            # B row: coefficients of voltage sources
            for j, vs in enumerate(voltage_symbols):
                B_matrix[i, j] = sp.simplify(rhs.coeff(vs))

        A_matrix = sp.simplify(A_matrix)
        B_matrix = sp.simplify(B_matrix)

        self.A_matrix = A_matrix
        self.B_matrix = B_matrix
        self.state_vars = state_sym
        self.voltage_symbols = voltage_symbols

        print("\n" + "=" * 70)
        print("STATE-SPACE REPRESENTATION: dq/dt = Aq + Bu")
        print("=" * 70)
        print("\nState vector q:")
        display(Latex(f"$q = {sp.latex(sp.Matrix(state_sym))}$"))
        print("\nInput vector u:")
        display(Latex(f"$u = {sp.latex(sp.Matrix(voltage_symbols))}$"))
        print("\nMatrix A:")
        display(Latex(f"$A = {sp.latex(A_matrix)}$"))
        print("\nMatrix B:")
        display(Latex(f"$B = {sp.latex(B_matrix)}$"))

        self._generate_matlab_file(A_matrix, B_matrix, state_sym, voltage_symbols)

    def _generate_matlab_file(self, A_matrix, B_matrix, state_vars, voltage_symbols):
        """Generate MATLAB .m file with symbolic A and B matrices"""
        lines = []
        lines.append("% State-Space Matrices for RC Circuit")
        lines.append("% System: dq/dt = A*q + B*u")
        lines.append("")
        lines.append(f"% State variables:  q = [{', '.join(str(v) for v in state_vars)}]'")
        lines.append(f"% Input variables:  u = [{', '.join(str(v) for v in voltage_symbols)}]'")
        lines.append("")
        lines.append("% Define symbolic variables")
        lines.append("syms a C real")
        lines.append("")

        def to_matlab(expr):
            s = sp.octave_code(expr)  # octave_code is compatible with MATLAB
            return s

        lines.append("% Matrix A")
        lines.append("A = [")
        for i in range(A_matrix.rows):
            row = ", ".join(to_matlab(A_matrix[i, j]) for j in range(A_matrix.cols))
            sep = ";" if i < A_matrix.rows - 1 else ""
            lines.append(f"    {row}{sep}")
        lines.append("];")
        lines.append("")

        lines.append("% Matrix B")
        lines.append("B = [")
        for i in range(B_matrix.rows):
            row = ", ".join(to_matlab(B_matrix[i, j]) for j in range(B_matrix.cols))
            sep = ";" if i < B_matrix.rows - 1 else ""
            lines.append(f"    {row}{sep}")
        lines.append("];")
        lines.append("")

        lines.append("% Display matrices")
        lines.append("disp('Matrix A:'); disp(A);")
        lines.append("disp('Matrix B:'); disp(B);")

        matlab_content = "\n".join(lines)
        with open('state_space_matrices.m', 'w', encoding='utf-8') as f:
            f.write(matlab_content)
        print("\n✅ MATLAB file 'state_space_matrices.m' generated successfully!")

    # ─────────────────────────────────────────────
    # Display Results
    # ─────────────────────────────────────────────

    def _display_results(self, q_orientation_map, branch_currents, q_map, kvl_equations,
                         kvl_equations_new_vars, solved_expressions, q_funcs, dq_funcs,
                         q_to_letter, q_symbols, dq_subs, branch_orientations):

        print("\nIndependent Currents and their chosen branches:")
        print("=" * 60)
        for q_name, branch_uv in q_orientation_map.items():
            letter = q_to_letter[q_name]
            print(f"Current on branch {branch_uv} (direction: {branch_uv[0]} → {branch_uv[1]}):")
            display(Latex(f"$i_{{({branch_uv[0]},{branch_uv[1]})}} = \\dot{{{letter}}}$"))

        print("\nDependent Currents in terms of independent ones:")
        print("=" * 60)
        for branch in self.all_branches:
            if branch not in q_map:
                expr = branch_currents[branch].subs(dq_subs)
                print(f"Current on branch {branch}:")
                display(Latex(f"$i_{{({branch[0]},{branch[1]})}} = {sp.latex(expr)}$"))

        print("\nKVL Differential Equations:")
        print("=" * 60)
        for i, eq in enumerate(kvl_equations):
            latex_eq = sp.latex(eq.lhs) + " = 0"
            for q_name, letter in q_to_letter.items():
                latex_eq = latex_eq.replace(f'{q_name}\\left(t\\right)', letter)
            display(Latex(f"\\text{{Loop {i+1}}}: \\quad {latex_eq}"))

        print("\nRewritten KVL Equations:")
        print("=" * 70)
        for i, eq in enumerate(kvl_equations_new_vars):
            latex_eq_new = sp.latex(eq.lhs) + " = 0"
            display(Latex(f"\\text{{Loop {i+1}}}: \\quad {latex_eq_new}"))

        print("\nSolving each equation for one derivative variable separately:")
        print("=" * 70)
        for i, sol_eq in enumerate(solved_expressions):
            if sol_eq is not None:
                display(Latex(f"\\text{{From Loop {i+1}}}: \\quad {sp.latex(sol_eq)}"))
            else:
                print(f"Loop {i+1}: Not solved.")

        self._generate_latex_output(
            q_orientation_map, branch_currents, q_map, kvl_equations,
            kvl_equations_new_vars, solved_expressions, q_to_letter,
            dq_subs, branch_orientations
        )

    # ─────────────────────────────────────────────
    # LaTeX Output
    # ─────────────────────────────────────────────

    def _generate_latex_output(self, q_orientation_map, branch_currents, q_map, kvl_equations,
                                kvl_equations_new_vars, solved_expressions, q_to_letter,
                                dq_subs, branch_orientations):

        independent_currents_latex = []
        for q_name, branch_uv in q_orientation_map.items():
            letter = q_to_letter[q_name]
            dir_text = f"\\text{{ (direction: {branch_uv[0]} $\\rightarrow$ {branch_uv[1]})}}"
            independent_currents_latex.append(
                f"i_{{({branch_uv[0]},{branch_uv[1]})}} &= \\dot{{{letter}}} \\quad {dir_text}"
            )

        dependent_currents_latex = []
        for branch in self.all_branches:
            if branch not in q_map:
                expr = branch_currents[branch].subs(dq_subs)
                dependent_currents_latex.append(
                    f"i_{{({branch[0]},{branch[1]})}} &= {sp.latex(expr)}"
                )

        latex_string_list = []
        for i, eq in enumerate(kvl_equations):
            latex_eq = sp.latex(eq.lhs) + " = 0"
            for q_name, letter in q_to_letter.items():
                latex_eq = latex_eq.replace(f'{q_name}\\left(t\\right)', letter)
            latex_string_list.append(f"\\text{{Loop {i+1}}}: & \\quad {latex_eq}")

        latex_new_eqs = []
        for i, eq in enumerate(kvl_equations_new_vars):
            latex_eq = sp.latex(eq.lhs) + " = 0"
            latex_new_eqs.append(f"\\text{{Loop {i+1}}}: & \\quad {latex_eq}")

        latex_solved = []
        for i, sol_eq in enumerate(solved_expressions):
            if sol_eq is not None:
                latex_solved.append(f"\\text{{From Loop {i+1}}}: & \\quad {sp.latex(sol_eq)}")
            else:
                latex_solved.append(f"\\text{{Loop {i+1}}}: & \\quad \\text{{Not solved}}")

        q_names_sorted = sorted(q_to_letter.keys(), key=lambda x: int(x.split('_')[1]))
        mapping_lines = [f"{q_name} &= {q_to_letter[q_name]}(t)" for q_name in q_names_sorted]

        A_latex = sp.latex(self.A_matrix)
        B_latex = sp.latex(self.B_matrix)
        state_vars_latex = sp.latex(sp.Matrix(self.state_vars))
        voltage_vars_latex = sp.latex(sp.Matrix(self.voltage_symbols))

        def align_block(lines):
            if lines:
                return r"\begin{align}" + "\n" + " \\\\\n".join(lines) + "\n" + r"\end{align}"
            return r"\begin{center}\textit{None}\end{center}"

        latex_equations_content = r"""
\documentclass[a4paper,12pt]{article}
\usepackage{amsmath,amsfonts,amssymb,geometry}
\usepackage[utf8]{inputenc}
\geometry{margin=1in}
\begin{document}
\title{KVL Differential Equations and State-Space Representation}
\date{\today}
\maketitle

\section{Circuit Parameters}
\begin{itemize}
\item Number of nodes: """ + str(self.num_nodes) + r"""
\item Number of branches: """ + str(self.N) + r"""
\item Number of capacitors: """ + str(self.num_capacitors) + r"""
\item Number of resistors: """ + str(len(self.resistor_branches)) + r"""
\item Independent currents needed: """ + str(self.num_independent) + r"""
\end{itemize}

\section{Variable Mapping}
We define new variables:
""" + align_block(mapping_lines) + r"""

\section{Independent Currents}
The following currents are chosen as independent variables:
""" + align_block(independent_currents_latex) + r"""

\section{Dependent Currents}
The following currents are expressed in terms of the independent currents:
""" + align_block(dependent_currents_latex) + r"""

\section{KVL Differential Equations}
The variable $a$ is defined as $a = RC$.
""" + align_block(latex_string_list) + r"""

\subsection{Rewritten KVL Equations}
""" + align_block(latex_new_eqs) + r"""

\section{Solved Expressions}
Each equation solved for one derivative variable:
""" + align_block(latex_solved) + r"""

\section{State-Space Representation}
\[
\frac{dq}{dt} = Aq + Bu
\]
where
\[
q = """ + state_vars_latex + r""", \quad u = """ + voltage_vars_latex + r"""
\]

\subsection{Matrix A}
\[
A = """ + A_latex + r"""
\]

\subsection{Matrix B}
\[
B = """ + B_latex + r"""
\]

\end{document}
"""
        with open('circuit_equations.tex', 'w', encoding='utf-8') as f:
            f.write(latex_equations_content)

        self._process_circuit_diagram(branch_currents, q_to_letter, branch_orientations, dq_subs, q_map)
        self._compile_and_display_files()

    def _process_circuit_diagram(self, branch_currents, q_to_letter, branch_orientations, dq_subs, q_map):
        try:
            with open('circuit_diagram.tex', 'r', encoding='utf-8') as f:
                circuit_content = f.read()

            circuit_content = circuit_content.replace('\\nnode', '\\node')
            circuit_content = circuit_content.replace('\\nodraw', '\\node')
            circuit_content = re.sub(r'\\ode\b', r'\\node', circuit_content)

            draw_pattern = r'\\draw\s*\(\s*(\d+)\s*\)\s*to\s*\[(.*?)\]\s*\(\s*(\d+)\s*\)\s*;'

            def replace_draw(match):
                from_coord, options, to_coord = match.groups()
                try:
                    u, v = int(from_coord), int(to_coord)
                except ValueError:
                    return match.group(0)
                branch = tuple(sorted((u, v)))
                if branch not in self.all_branches:
                    return match.group(0)
                current_expr = branch_currents[branch].subs(dq_subs)
                current_display = sp.latex(current_expr).replace('\\cdot', '').replace(' ', '')
                latex_current = f"\\footnotesize ${current_display}$"
                orient = branch_orientations[branch]
                if 'i^>=' in options or 'i_<=' in options:
                    return match.group(0)
                label_pos = f", i^>={latex_current}" if (u, v) == orient else f", i_<={latex_current}"
                return f"\\draw ({from_coord}) to[{options}{label_pos}] ({to_coord});"

            modified_content = re.sub(draw_pattern, replace_draw, circuit_content, flags=re.DOTALL)

            caption_parts = []
            for q_name in sorted(q_to_letter.keys(), key=lambda x: int(x.split('_')[1])):
                letter = q_to_letter[q_name]
                num = q_name.split('_')[1]
                caption_parts.append(f"$q_{{{num}}} \\rightarrow {letter}$")
            caption_text = "Variable mapping: " + ", ".join(caption_parts) + "."

            circuit_block_match = re.search(r'(\\begin{circuitikz}.*?\\end{circuitikz})',
                                             modified_content, re.DOTALL)
            if circuit_block_match:
                circuit_block = circuit_block_match.group(1)
                final_circuit_latex = f"""
\\documentclass[a4paper,12pt]{{article}}
\\usepackage{{amsmath,geometry,circuitikz,caption}}
\\usepackage[utf8]{{inputenc}}
\\usetikzlibrary{{calc,positioning}}
\\geometry{{margin=1in}}
\\begin{{document}}
\\begin{{figure}}[h!]
\\centering
{circuit_block}
\\caption*{{{caption_text}}}
\\end{{figure}}
\\end{{document}}
"""
            else:
                print("⚠️ Could not find circuitikz block.")
                final_circuit_latex = f"""
\\documentclass[a4paper,12pt]{{article}}
\\usepackage{{amsmath,geometry,circuitikz}}
\\usepackage[utf8]{{inputenc}}
\\geometry{{margin=1in}}
\\begin{{document}}
\\begin{{figure}}[h!]
\\centering
\\begin{{circuitikz}}
\\node[draw, text width=10cm, align=center] {{Diagram could not be processed. {caption_text}}};
\\end{{circuitikz}}
\\end{{figure}}
\\end{{document}}
"""
            with open('circuit_diagram.tex', 'w', encoding='utf-8') as f:
                f.write(final_circuit_latex)

        except FileNotFoundError:
            print("⚠️ Circuit diagram file not found. Creating default.")
            self._create_default_diagram()
        except Exception as e:
            print(f"⚠️ Error processing circuit diagram: {e}")
            self._create_default_diagram()

    def _compile_and_display_files(self):
        def compile_latex(filename):
            try:
                for _ in range(2):
                    subprocess.run(['pdflatex', '-interaction=nonstopmode', filename],
                                   check=True, capture_output=True, text=True)
                print(f"✅ {filename} compiled successfully")
                return True
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Error compiling {filename}:")
                for line in e.stdout.split('\n'):
                    if line.startswith('!'):
                        print(f"  LaTeX Error: {line}")
                return False
            except FileNotFoundError:
                print("⚠️ pdflatex not found. Please install LaTeX.")
                return False

        compile_latex('circuit_equations.tex')
        compile_latex('circuit_diagram.tex')

        available_files = []
        for fname, desc, icon in [
            ('circuit_equations.pdf', 'Equations PDF', '📄'),
            ('circuit_diagram.pdf', 'Circuit Diagram PDF', '🔌'),
            ('state_space_matrices.m', 'MATLAB State-Space File', '📊'),
        ]:
            if os.path.exists(fname):
                available_files.append((fname, desc, icon))

        if available_files:
            buttons = []
            for fname, desc, icon in available_files:
                btn = widgets.Button(
                    description=f"{icon} Download {desc}",
                    button_style='info',
                    layout=widgets.Layout(width='350px', margin='5px')
                )
                def make_handler(f):
                    def on_click(b):
                        files.download(f)
                    return on_click
                btn.on_click(make_handler(fname))
                buttons.append(btn)
            display(widgets.VBox([
                widgets.HTML("<h3>📥 Click to download files:</h3>"),
                widgets.VBox(buttons)
            ]))
        else:
            print("\n❌ No files available for download. Check compilation errors.")

    # ─────────────────────────────────────────────
    # Run
    # ─────────────────────────────────────────────

    def run(self):
        self.create_widgets()
        display(widgets.VBox(self.dropdowns + [self.calculate_button]), self.output)