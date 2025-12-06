import os
import re
import random
import subprocess
import numpy as np
from textwrap import dedent

# ================= USER SETTINGS =================
BASE_DATA   = "perfect_512.dat"   # base Si data file (512 atoms, at T_REF)
SI_SW_FILE  = "Si.sw"             # Stillinger–Weber potential file

T_REF       = 1000.0              # reference T (K) of BASE_DATA (used in scaling)
TEMPS       = [700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0]
N_C_LIST    = [1, 5, 10, 20]      # number of C contaminants for each case

RANDOM_SEED = 485                 
LAMMPS_EXE  = "lmp"               
# ==================================================


def alpha_Si(T):
    """
    Linear thermal expansion coefficient alpha(T) [1/K] for Si.
    T can be scalar or array.
    """
    T = np.asarray(T, dtype=float)
    term = 3.725 * (1.0 - np.exp(-5.8e-3 * (T - 124.0)))
    term += 5.548e-4 * T
    return term * 1.0e-6  # 1/K


def length_scale(T, T_ref=T_REF):
    if T == T_ref:
        return 1.0
    n_steps = int(abs(T - T_ref)) or 1
    Ts = np.linspace(T_ref, T, n_steps + 1)
    alphas = alpha_Si(Ts)
    integral = np.trapz(alphas, Ts)
    return 1.0 + integral


def make_scaled_doped_data(base_path, out_path, T, n_C, seed=None):

    if seed is not None:
        random.seed(seed)

    with open(base_path, "r") as f:
        lines = f.read().splitlines()

    atom_types_idx = None
    for i, line in enumerate(lines):
        if "atom types" in line:
            atom_types_idx = i
            parts = line.split()
            ntypes = int(parts[0])
            if ntypes < 2:
                parts[0] = "2"
                lines[i] = " ".join(parts)
            break
    if atom_types_idx is None:
        raise RuntimeError("Could not find 'atom types' line in data file.")

    mass_header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Masses"):
            mass_header_idx = i
            break
    if mass_header_idx is None:
        raise RuntimeError("Cannot find 'Masses' section.")

    # Find the first mass line after 'Masses'
    j = mass_header_idx + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1
    first_mass_idx = j

    have_mass2 = False
    k = first_mass_idx
    # Mass lines look like: "<int> <float> ..."
    while k < len(lines) and re.match(r"^\s*\d+\s", lines[k]):
        if lines[k].split()[0] == "2":
            have_mass2 = True
        k += 1
    end_mass_idx = k

    if not have_mass2:
        # Insert mass for type 2 just after existing masses
        lines.insert(end_mass_idx, "2 12.011    # C")

    xline_idx = yline_idx = zline_idx = None
    atoms_line_idx = None

    for i, line in enumerate(lines):
        if "xlo xhi" in line:
            xline_idx = i
        elif "ylo yhi" in line:
            yline_idx = i
        elif "zlo zhi" in line:
            zline_idx = i
        elif line.strip().startswith("Atoms"):
            atoms_line_idx = i
            break

    if None in (xline_idx, yline_idx, zline_idx, atoms_line_idx):
        raise RuntimeError("Could not locate box bounds or 'Atoms' section.")

    def parse_bounds(idx):
        p = lines[idx].split()
        return float(p[0]), float(p[1])

    xlo, xhi = parse_bounds(xline_idx)
    Lx = xhi - xlo

    scale = length_scale(T)
    print(f"    Scaling to T={T} K -> scale = {scale:.6e}  (L_new ≈ {Lx*scale:.4f} Å)")

    def scale_bounds(idx):
        p = lines[idx].split()
        lo = float(p[0])
        hi = float(p[1])
        new_hi = lo + (hi - lo) * scale
        tail = " ".join(p[2:])
        lines[idx] = f"{lo:.8f} {new_hi:.8f} {tail}"

    scale_bounds(xline_idx)
    scale_bounds(yline_idx)
    scale_bounds(zline_idx)

    atoms_start = atoms_line_idx + 2  # skip "Atoms # atomic" and blank line
    atoms_end = None
    for j in range(atoms_start, len(lines)):
        s = lines[j].strip()
        if not s:
            atoms_end = j
            break
        if s[0].isalpha():  # next section header
            atoms_end = j
            break
    if atoms_end is None:
        atoms_end = len(lines)

    atom_line_indices = list(range(atoms_start, atoms_end))
    N = len(atom_line_indices)
    if n_C > N:
        raise ValueError(f"Requested {n_C} C atoms but only {N} atoms exist in data file.")
        
    c_indices = set(random.sample(atom_line_indices, n_C))

    for idx in atom_line_indices:
        parts = lines[idx].split()
        if len(parts) < 5:
            continue  # skip weird lines just in case
        atype = int(parts[1])
        x, y, z = map(float, parts[2:5])
        # scale coordinates with box
        x *= scale
        y *= scale
        z *= scale
        if idx in c_indices:
            atype = 2  # mark as C
        parts[1] = str(atype)
        parts[2:5] = [f"{x:.8f}", f"{y:.8f}", f"{z:.8f}"]
        lines[idx] = " ".join(parts)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_lammps_input(out_path, data_filename, T, n_C, tag):
    """
    Write a run_lammps_*.in file for a given case (T, n_C).
    """
    txt = f"""
    # C contaminant run: {tag} (T={T} K, N_C={n_C})

    units           metal
    atom_style      atomic
    boundary        p p p

    read_data       {data_filename}

    mass            1 28.0855
    mass            2 12.011

    pair_style      sw
    pair_coeff      * * {SI_SW_FILE} Si Si

    neighbor        2.0 bin
    neigh_modify    delay 0 every 1 check yes

    timestep        0.001    # 1 fs

    velocity        all create {T} 12345 mom yes rot yes dist gaussian

    fix             f1 all nvt temp {T} {T} $(100.0*dt)

    thermo_style    custom step temp pe ke etotal
    thermo          10000

    dump            d1 all custom 100 state_dump_{tag} id type x y z vx vy vz
    dump_modify     d1 sort id

    run             10000000   # 10 ns

    write_data      FINAL_STRUCTURE_{tag}
    """
    with open(out_path, "w") as f:
        f.write(dedent(txt))


def main():
    if not os.path.isfile(BASE_DATA):
        raise FileNotFoundError(f"Base data file '{BASE_DATA}' not found.")
    if not os.path.isfile(SI_SW_FILE):
        raise FileNotFoundError(f"Potential file '{SI_SW_FILE}' not found.")

    print("=== Building and RUNNING all C-contaminant cases ===")
    print(f"  Base data : {BASE_DATA} (T_ref = {T_REF} K)")
    print(f"  SW file   : {SI_SW_FILE}")
    print(f"  LAMMPS    : {LAMMPS_EXE}")
    print()

    case_id = 0
    for T in TEMPS:
        for n_C in N_C_LIST:
            case_id += 1
            tag = f"T{int(T)}K_C{n_C}"
            out_data = f"starting_config_{tag}.dat"
            in_file  = f"run_lammps_{tag}.in"

            print(f"--- Case {case_id}: {tag} ---")
            print(f"    T   = {T} K")
            print(f"    N_C = {n_C}")

            seed = RANDOM_SEED + int(T) * 100 + n_C

            # Build data file
            make_scaled_doped_data(
                BASE_DATA,
                out_data,
                T,
                n_C,
                seed=seed,
            )

            # Write LAMMPS input
            write_lammps_input(in_file, out_data, T, n_C, tag)

            print(f"    -> wrote data:  {out_data}")
            print(f"    -> wrote input: {in_file}")

            # Run LAMMPS for this case
            cmd = [LAMMPS_EXE, "-in", in_file]
            print(f"    >>> Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f"    !!! LAMMPS run FAILED for {tag} (exit code {result.returncode})")
                # You can choose to break or raise here if you want to stop on failure
                # raise RuntimeError(f"LAMMPS failed for {tag}")
            else:
                print(f"    <<< LAMMPS run finished OK for {tag}")
            print()

    print("=== All cases attempted. Check output files and logs. ===")


if __name__ == "__main__":
    main()

