import numpy 
from scipy.special import legendre
import re
import matplotlib.pyplot as plt
import os

def theta_vals(params):

    xy = params["xy"] # point at centroid of elements 

    n_layers = params["N_layers"]
    Lx, Ly = params["Geometry"]

    st_sq = numpy.deg2rad(params["st_sq"])
    # st_sq = params["st_sq"]
    var_type = params["var_type"]

    th = numpy.zeros((n_layers,len(xy[0,:])))

    for l in range(n_layers):
        
        anglexy = st_sq[l,:]  # T0, T1
        matrix = numpy.zeros((len(anglexy), len(anglexy)))

        if var_type == 'x':          # csi variation [-1,1]
            xy_points = xy[0,:]
            points_l = numpy.linspace(-1, 1,len(anglexy))   # point of T0, T1
            points = ((xy_points*(2)/Lx - 1))                         # points normalization [-1,1]
        elif var_type == '|x|':                # csi variation [0,1]
            xy_points = xy[0,:]
            points_l = numpy.linspace(0, 1,len(anglexy))   # point of T0, T1
            points = ((xy_points*(2)/Lx - 1))                         # points normalization [-1,1]
        
        if var_type == 'y':          # eta variation [-1,1]
            xy_points = xy[1,:]
            points_l = numpy.linspace(-1, 1,len(anglexy))   # point of T0, T1
            points = ((xy_points*(2)/Ly - 1))                         # points normalization [-1,1]
        elif var_type == '|y|':                # eta variation [0,1]
            xy_points = xy[1,:]
            points_l = numpy.linspace(0, 1,len(anglexy))   # point of T0, T1
            points = ((xy_points*(2)/Ly - 1))                         # points normalization [-1,1]
        
        for j in range(len(anglexy)):
                    matrix[:,j] = legendre(j)(points_l)
        
        coeff_pol = numpy.linalg.inv(matrix)@anglexy
        pol_matrix = numpy.zeros((len(xy_points), len(anglexy)))
        for j in range(len(anglexy)):
            pol_matrix[:,j] = legendre(j)(points)

        angle = pol_matrix@(coeff_pol)

        th[l,:] = angle

    return th

def generate_centers_old(Lx, Ly, Nx, Ny):
    dx = Lx / Nx
    dy = Ly / Ny
    x_centers = numpy.linspace(dx/2, Lx - dx/2, Nx)
    y_centers = numpy.linspace(dy/2, Ly - dy/2, Ny)
    return x_centers, y_centers


def generate_centers(Lx, Ly, Nx, Ny, sweep=0):
   
    dx = Lx / Nx
    dy = Ly / Ny
    sweep_offset = Ly * numpy.tan(sweep)

    y_centers = numpy.linspace(dy/2, Ly - dy/2, Ny)
    x_centers = numpy.linspace(dx/2, Lx - dx/2, Nx)

    x_centers_shifted = numpy.zeros((Ny, Nx))
    for j, y_c in enumerate(y_centers):
        x_offset = sweep_offset * (y_c / Ly)
        x_centers_shifted[j, :] = x_centers + x_offset

    return x_centers_shifted, y_centers

def plot_nast(filename):
    file_path = filename+".f06"
    # Numero massimo di Mode da plottare
    max_modes_to_plot = 4  # aggiorna questo valore secondo le tue necessità
    trsh = 1e-2

    # Definisci i marcatori di inizio e fine della tabella
    mode_marker = "POINT ="
    start_marker = "KFREQ            1./KFREQ         VELOCITY            DAMPING         FREQUENCY            COMPLEX   EIGENVALUE"
    end_marker = "MSC.NASTRAN"

    # Leggi il contenuto del file
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Estrai le righe tra i marcatori e accumula dati per ogni mode
    mode_data = {}
    current_mode = None
    capture_data = False
    current_data = []

    for line in content:
        if mode_marker in line:
            match = re.search(r"POINT\s*=\s*(\d+)", line)
            if match:
                new_mode = int(match.group(1))
                if current_mode is not None and current_data:
                    if current_mode not in mode_data:
                        mode_data[current_mode] = []
                    mode_data[current_mode].extend(current_data)
                current_mode = new_mode
                current_data = []
            capture_data = False
        if start_marker in line:
            capture_data = True
            continue  # Salta la riga con il marcatore di inizio
        
        if capture_data:
            if end_marker in line:
                capture_data = False
                if current_mode is not None and current_data:
                    if current_mode not in mode_data:
                        mode_data[current_mode] = []
                    mode_data[current_mode].extend(current_data)
                current_data = []
            else:
                current_data.append(line.strip())

    # Se c'è una sezione di dati non ancora salvata, la aggiungiamo
    if current_mode is not None and current_data:
        if current_mode not in mode_data:
            mode_data[current_mode] = []
        mode_data[current_mode].extend(current_data)

    # Converti ogni sezione in una matrice numpy e processa i dati
    data_arrays = []
    expected_columns = 7

    for mode in sorted(mode_data.keys())[:max_modes_to_plot]:
        data_matrix = []
        for line in mode_data[mode]:
            numbers = [num for num in re.split(r'\s+', line) if num]
            if len(numbers) == expected_columns:
                row = []
                for num in numbers:
                    try:
                        row.append(float(num))
                    except ValueError:
                        row.append(numpy.nan)
                data_matrix.append(row)
        if data_matrix:
            data_array = numpy.array(data_matrix)
            # Ordina per VELOCITY (colonna 2) e rimuovi i duplicati
            data_array = data_array[numpy.argsort(data_array[:, 2])]
            _, unique_indices = numpy.unique(data_array[:, 2], return_index=True)
            data_array = data_array[unique_indices]
            data_arrays.append((mode, data_array))

    # Genera i grafici
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    # fig1, axes1 = plt.subplots(2, 1, figsize=(10, 12))

    # Variabili per la tabella finale
    modes = []
    velocities = []
    dampings = []
    frequencies = []

    # Variabili per la tabella di interpolazione
    interpolated_modes = []
    interpolated_velocities = []
    interpolated_frequencies = []

    # Funzione di interpolazione lineare
    def interpolate_zero_damping(x1, y1, x2, y2, y=0):
        m = (y1-y2)/(x1-x2)
        q = y1- m*x1
        x = (y-q)/m
        return x

    def interpolate_freq(x1, y1, x2, y2, x):
        m = (y1-y2)/(x1-x2)
        q = y1- m*x1
        y = m*x+q
        return y

    # Primo grafico: Velocity vs Nuova Variabile
    for mode, data_array in data_arrays:
        # omega = numpy.sqrt(data_array[:, 5]**2 + data_array[:, 6]**2)
        omega = data_array[:, 6]
        axes[0].plot(data_array[:, 2], omega, marker='o', linestyle='-', label=f'Mode {mode}')
        # omega = numpy.sqrt(data_array[:, 5]**2 + data_array[:, 6]**2)
        # axes1[0].plot(data_array[:, 2], omega, marker='o', linestyle='-', label=f'Mode {mode}')
        
        # Trova la velocità per cui il damping è maggiore di un threshold

        positive_damping_indices = numpy.where(data_array[:, 3] > trsh)[0]
        if positive_damping_indices.size > 0:
            first_positive_index = positive_damping_indices[0]
            if first_positive_index > 0:
                v1 = data_array[first_positive_index - 1, 2]
                w1 = data_array[first_positive_index - 1, 6]
                d1 = data_array[first_positive_index - 1, 3]
                v2 = data_array[first_positive_index, 2]
                w2 = data_array[first_positive_index, 6]
                d2 = data_array[first_positive_index, 3]
                zero_damping_velocity = interpolate_zero_damping(v1, d1, v2, d2)
                zero_damping_frequency = interpolate_freq(v1, w1, v2, w2, zero_damping_velocity)
                interpolated_modes.append(mode)
                interpolated_velocities.append(zero_damping_velocity)
                interpolated_frequencies.append(zero_damping_frequency)
            modes.append(mode)
            velocities.append(data_array[first_positive_index, 2])
            dampings.append(data_array[first_positive_index, 3])
            frequencies.append(data_array[first_positive_index, 6])
        else:
            modes.append(mode)
            velocities.append(numpy.nan)
            dampings.append(numpy.nan)
            frequencies.append(numpy.nan)

    axes[0].set_xlabel("Velocity")
    axes[0].set_ylabel("Omega")
    axes[0].set_title("Velocity vs Omega for each Mode")
    # axes1[0].set_xlabel("Velocity")
    # axes1[0].set_ylabel("|Omega|")
    # axes1[0].set_title("Velocity vs |Omega| for each Mode")

    # Secondo grafico: Velocity vs Damping
    for mode, data_array in data_arrays:
        # axes[1].plot(data_array[:, 2], data_array[:, 3], marker='o', linestyle='-', label=f'Mode {mode}')
        sigma =  data_array[:, 5]
        axes[1].plot(data_array[:, 2],sigma, marker='o', linestyle='-', label=f'Mode {mode}')
        # damping = -sigma/(numpy.sqrt(data_array[:, 5]**2 + data_array[:, 6]**2))
        # axes1[1].plot(data_array[:, 2],damping, marker='o', linestyle='-', label=f'Mode {mode}')

    axes[1].set_xlabel("Velocity")
    axes[1].set_ylabel("sigma")
    axes[1].set_title("Velocity vs sigma for each Mode")
    # axes1[1].set_xlabel("Velocity")
    # axes1[1].set_ylabel("Damping")
    # axes1[1].set_title("Velocity vs Damping for each Mode")

    # Aggiungi una legenda condivisa al di fuori dell'area di plot
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)



    # Crea la tabella finale
    table_data = numpy.column_stack((modes, velocities, dampings, frequencies))
    table_header = ["Mode", "Velocity with Damping > 0", "Sigma", "Omega"]

    # Stampa la tabella finale
    print(f"{table_header[0]:<10} {table_header[1]:<30} {table_header[2]:<15} {table_header[3]:<15}")
    for row in table_data:
        mode, velocity, damping, frequency = row
        velocity_str = f"{velocity:.2f}" if not numpy.isnan(velocity) else ""
        damping_str = f"{damping:.2f}" if not numpy.isnan(damping) else ""
        frequency_str = f"{frequency:.2f}" if not numpy.isnan(frequency) else ""
        print(f"{int(mode):<10} {velocity_str:<30} {damping_str:<15}  {frequency_str:<15}")

    # Crea la tabella di interpolazione
    interpolation_table_header = ["Mode", "Interpolated Velocity", "Interpolated frequency"]

    print("\nInterpolated Velocities and frequencies:")
    print(f"{interpolation_table_header[0]:<10} {interpolation_table_header[1]:<30} {interpolation_table_header[2]:<15}")
    for mode, velocity, frequency in zip(interpolated_modes, interpolated_velocities, interpolated_frequencies):
        velocity_str = f"{velocity:.4f}" if not numpy.isnan(velocity) else ""
        frequency_str = f"{frequency:.8f}" if not numpy.isnan(frequency) else ""
        print(f"{int(mode):<10} {velocity_str:<30} {frequency_str:<15}")

    plt.tight_layout()
    if os.name == 'nt':
        plt.show()
    elif os.name == 'posix':
        for fmt in ["png"]:
            fig.savefig("Figures/Flt."+fmt, format = fmt, bbox_inches = 'tight')




def print_div(filename):
    file_path = filename+".f06"
    with open(file_path, 'r') as file:
        lines = file.readlines()

    dynamic_pressures = []
    in_divergence_section = False

    for i, line in enumerate(lines):
        if "D I V E R G E N C E      S U M M A R Y" in line:
                in_divergence_section = True
                continue

        elif in_divergence_section:
            if "*** USER INFORMATION MESSAGE" in line:
                break

            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    dp = float(parts[1])
                    dynamic_pressures.append(dp)
                except ValueError:
                    continue  

    return numpy.array(dynamic_pressures)

def print_disp_SAA(filename, idLE = 4, idTE = 3):
    file_path = filename+".f06"

    in_disp_section = False
    uz_tip_LE = None
    uz_tip_TE = None
    with open(file_path, 'r') as file:
        for line in file:
            if "D I S P L A C E M E N T   V E C T O R" in line:
                in_disp_section = True
                continue

            if in_disp_section and "POINT ID." in line and "T3" in line:
                continue

            if in_disp_section:
                parts = line.strip().split()
                if len(parts) >= 6 and parts[0].isdigit():
                    try:
                        point_id = int(parts[0])
                        uz_val = float(parts[4])

                        if point_id == idTE and uz_tip_TE is None:
                            uz_tip_TE = uz_val
                        elif point_id == idLE and uz_tip_LE is None:
                            uz_tip_LE = uz_val
                    except ValueError:
                        continue

                elif "1    SSA OF COMPOSITE PANEL" in line: # not work... but it is ok right now
                    break

    return uz_tip_LE, uz_tip_TE

def print_vib(filename):
    file_path = filename+".f06"
    res = []
    in_eigenvalue_section = False

    with open(file_path, 'r') as file:
        for line in file:
            if "R E A L   E I G E N V A L U E S" in line:
                in_eigenvalue_section = True
                continue

            elif in_eigenvalue_section:
                parts = line.strip().split()

                if len(parts) >= 6:
                    try:
                        lam = float(parts[2])
                        w = float(parts[3])
                        f = float(parts[4])
                        res.append((lam, w, f))
                    except ValueError:
                        continue

                elif "MSC.NASTRAN" in line or line.strip() == "":
                    break

    return res
