import numpy as np
import utilities
import subprocess
import os
import sys

import gmsh

from pyNastran.converters.nastran.nastran_to_vtk import nastran_to_vtk



# WHICH TEST
if (len(sys.argv) < 3):
    print("Please specify test data and analysis type")
    quit()

TEST = sys.argv[1]
ANALYSIS = sys.argv[2]

# TEST = "T1"
# ANALYSIS = "DIV"

# Solver -- chose one
if ANALYSIS == "FVIB":
    sol, filename = 103, "panelFVIB"          # FVIB
elif ANALYSIS == "DIV":
    sol, filename = 144, "panelDIV"           # DIV
elif ANALYSIS == "FLT":
    sol, filename = 145, "panelFLT"           # FLT
else:
    print("Analysis type not yet implemented. Choose between FVIB, DIV and FLT  ")
    quit()

# params = {}
# # params["st_sq"] = np.array([[90, 90],[90,90],[0,0],[0,0],[90,90],[90,90]])
# # params["st_sq"] = np.array([[0, 90],[90,0],[90,0],[0,90]])
# params["st_sq"] = np.array([[90, 45],[90,-45],[45,90],[-45,90],[-45,90],[45,90],[90,-45],[90,45]])
# params["var_type"] = '|y|'                        # x or  y -->  T0 @ -1 and T1 @ 1 --------- |x| or |y| T0 @ 0 and T1 @ 1
# params["N_layers"] = params["st_sq"].shape[0]
# t_ply = 0.134e-3
# thickness = params["N_layers"] * t_ply
# thickness = 0.804e-3  # Spessore totale 


if TEST == "T1":
    params = {}
    params["st_sq"] = np.array([[90, -90]])
    params["var_type"] = 'y'                        # x or  y -->  T0 @ -1 and T1 @ 1 --------- |x| or |y| T0 @ 0 and T1 @ 1
    params["N_layers"] = params["st_sq"].shape[0]
    t_ply = 0.134e-3
    thickness = params["N_layers"] * t_ply
    # Lato da vincolare: 'left', 'right', 'top', 'bottom'
    side_to_fix = 'bottom'
elif TEST == "T2":
    params = {}
    params["st_sq"] = np.array([[0, 90]])
    params["var_type"] = 'y'                        # x or  y -->  T0 @ -1 and T1 @ 1 --------- |x| or |y| T0 @ 0 and T1 @ 1
    params["N_layers"] = params["st_sq"].shape[0]
    t_ply = 0.134e-3
    thickness = params["N_layers"] * t_ply
    # Lato da vincolare: 'left', 'right', 'top', 'bottom'
    side_to_fix = 'bottom'
elif TEST == "T3":
    params = {}
    params["st_sq"] = np.array([[45, -45]])
    params["var_type"] = 'y'                        # x or  y -->  T0 @ -1 and T1 @ 1 --------- |x| or |y| T0 @ 0 and T1 @ 1
    params["N_layers"] = params["st_sq"].shape[0]
    t_ply = 0.134e-3
    thickness = params["N_layers"] * t_ply
    # Lato da vincolare: 'left', 'right', 'top', 'bottom'
    side_to_fix = 'bottom'
elif TEST == "T4":
    params = {}
    params["st_sq"] = np.array([[-60, 30]])
    params["var_type"] = 'y'                        # x or  y -->  T0 @ -1 and T1 @ 1 --------- |x| or |y| T0 @ 0 and T1 @ 1
    params["N_layers"] = params["st_sq"].shape[0]
    t_ply = 0.134e-3
    thickness = params["N_layers"] * t_ply
    # Lato da vincolare: 'left', 'right', 'top', 'bottom'
    side_to_fix = 'bottom'
else:
    print("Test non definito")
    quit()

# COMMON PROPS FOR T1-->T4
# Material
E1 = 98.0e9
E2 = 7.9e9
G12 = 5.6e9
G23 = 5.6e9
G32 = 5.6e9
nu12 = 0.28
rho = 1520

props = [E1, E2, nu12, G12, G23, G32, rho]

# === Geometry
Lx = 0.0762
Ly = 0.305
Nx = 16  # ELEMENTI lungo x
Ny = 64   #  ELEMENTI lungo y
Nelems = Nx*Ny
sweep = np.deg2rad(0)

## AERO
rho_f = 1.226

# SAA
alpha = 1
V = 0.5
q = 0.5*rho_f*V**2
Nroots = 10

# FLT
V_range = np.linspace(1,100,10)
V_plot = -10                    # TODO, add negative velocities if eigenvector are requested


## STRUCTURE
gmsh.initialize()
gmsh.model.add("mesh")



p1 = gmsh.model.geo.addPoint(0, 0, 0, 0)
p2 = gmsh.model.geo.addPoint(Lx, 0, 0, 0)
p3 = gmsh.model.geo.addPoint(Ly*np.tan(sweep)+Lx, Ly, 0, 0)
p4 = gmsh.model.geo.addPoint(Ly*np.tan(sweep), Ly, 0, 0)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
ps = gmsh.model.geo.addPlaneSurface([cl])

gmsh.model.geo.synchronize()

# === Mesh
gmsh.model.mesh.setTransfiniteSurface(ps)
gmsh.model.mesh.setTransfiniteCurve(l1, Nx + 1)
gmsh.model.mesh.setTransfiniteCurve(l3, Nx + 1)
gmsh.model.mesh.setTransfiniteCurve(l2, Ny + 1)
gmsh.model.mesh.setTransfiniteCurve(l4, Ny + 1)
gmsh.model.mesh.setRecombine(2, ps)  # mesh quadrangolare

gmsh.model.mesh.generate(2)

node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
nodes = {}
elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
# renumbering of elements
gmsh.model.mesh.renumberElements(elem_tags[0],[i+1 for i in range(0,len(elem_tags[0]))])
elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)


## FLT SETUP
k_freq = np.linspace(0.001,5,20)
Mach = np.zeros_like(k_freq)
MK = np.empty(40)
MK[0::2] = Mach
MK[1::2] = k_freq
MK = MK.reshape(5, 8)



# OUTPUT
cwd = os.getcwd()
filepath = cwd+"/"+filename


## OTHER SETTINGS
run = True
plot = True
save_mesh = True
save_vtk = True
# result type
hdf5 = False
xdb = False
op2 = True

if save_mesh: gmsh.write(filename+".msh")

print()
print("#######################################")
print("STARTING JOB: ",filename)
print("NASTRAN STR DOF: ", (Nx+1)*(Ny+1)*6)
print("NASTRAN AERO DOF: ", (Nx)*(Ny))
print("NASTRAN DOF: ", (Nx)*(Ny)+(Nx+1)*(Ny+1)*6)
print("#######################################")
print()

## Compute theta for VAT laminate
centers = gmsh.model.mesh.getBarycenters(3,1,0,0).reshape(Nelems, 3)
x_centers, y_centers, z_centers = centers.T
X, Y = np.meshgrid(x_centers,y_centers)
Z = np.zeros_like(X)

params["xy"] = np.vstack((x_centers, y_centers, z_centers))
params["Geometry"] = np.array([Lx,Ly])

theta = np.rad2deg(utilities.theta_vals(params))

bdf_filename = filename + '.bdf'


with open(bdf_filename, "w") as f:
    # === Intestazione ===
    f.write("$ MSC.Nastran input file generated by script\n")
    f.write("SOL {:d}\n".format(sol))
    f.write("CEND\n")
    
    ## Direct text input section
    if sol == 103:
        f.write("TITLE = Modal analysis of composite panel\n")
        f.write("ECHO = NONE\n")
        f.write("SUBCASE 1\n")
        f.write("   SUBTITLE = Default\n")
        f.write("   METHOD = 1\n")
        f.write("   SPC = 2\n")
        f.write("   VECTOR(SORT1,REAL)=ALL\n")
        f.write("   SPCFORCES(SORT1,REAL)=ALL\n")
    elif sol == 144:
        f.write("DIVERG = 50\n")
        f.write("CMETHOD = 50\n")
        f.write("TITLE = SSA of composite panel\n")
        f.write("ECHO = NONE\n")
        f.write("AECONFIG = AeroSG2D\n")
        f.write("SUBCASE 1\n")
        f.write("   SUBTITLE = Default\n")
        f.write("   SPC = 2\n")
        f.write("   DISPLACEMENT(SORT1,REAL)=ALL\n")
        f.write("   SPCFORCES(SORT1,REAL)=ALL\n")
        f.write("   STRESS(SORT1,REAL,VONMISES,BILIN)=ALL\n")
        f.write("TRIM = 1\n")
        f.write("AESYMXZ = Symmetric\n")
        f.write("AESYMXY = Asymmetric\n")
        f.write("AEROF = ALL\n")
        f.write("APRES = ALL\n")
    elif sol == 145:
        f.write("TITLE = Flutter analysis of composite panel\n")
        f.write("ECHO = NONE\n")
        f.write("AECONFIG = AeroSG2D\n")
        f.write("SUBCASE 1\n")
        f.write("   SUBTITLE = Default\n")
        f.write("   METHOD = 1\n")
        f.write("   SPC = 2\n")
        f.write("   VECTOR(SORT1,REAL)=ALL\n")
        f.write("   SPCFORCES(SORT1,REAL)=ALL\n")
        f.write("FMETHOD = 1\n")
        f.write("AESYMXZ = Symmetric\n")
        f.write("AESYMXY = Asymmetric\n")


    f.write("BEGIN BULK\n")

    # Parametri
    if hdf5:
        f.write("HDF5OUT PRCISION 32     CMPRMTHD LZ4     LEVEL   5\n")
    elif xdb:
        f.write("PARAM    POST      0\n")
    elif op2:
        op2_filename = filename + '.op2'
        f.write("PARAM    POST      1\n")

        
    f.write("PARAM   PRTMAXIM YES\n")
    if sol == 103 or sol == 145:
        f.write("EIGRL    1                       10      0\n")
    elif sol == 144 or sol ==145:
        f.write("PARAM    WTMASS 1.\n")
        f.write("PARAM    SNORM  20.\n")


    # === PCOMP ===
    t_ply = thickness / len(params["st_sq"])
    pcomp_id = [int(i) for i in elem_tags[0]]
    for pcomp in pcomp_id:
        f.write("PCOMP    {:d}\n".format(pcomp))
        laminate_stack = theta[:,pcomp-1]
        for angle in laminate_stack:
            t_formatted = "{:.7e}".format(t_ply).replace("e-0", "-").replace("e+0", "+").replace("e-","-").replace("e+","+")
            f.write("*        1               {:<16}{:<14f}  YES\n".format(t_formatted, angle))
        f.write("*\n")

    # === Mesh ===
    # dx = Lx / Nx
    # dy = Ly / Ny
    # node_id = 1
    # nodes_old = {}
    # for j in range(Ny+1):
    #     for i in range(Nx+1):
    #         x = i * dx
    #         y = j * dy
    #         z = 0.0
    #         f.write("GRID,{:d},,{:<8.5f},{:<8.5f},{:<8.5f}\n".format(node_id, x, y, z))
    #         nodes_old[(i, j)] = node_id
    #         node_id += 1
    
    for i, tag in enumerate(node_tags):
        x = node_coords[3*i]
        y = node_coords[3*i + 1]
        z = node_coords[3*i + 2]
        f.write("GRID,{:d},,{:<8.5f},{:<8.5f},{:<8.5f}\n".format(tag, x, y, z))
        nodes[tag] = (x, y, z)
    
    # === Elementi ===

    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if gmsh.model.mesh.getElementProperties(etype)[0] == "Quadrilateral 4":
            for eid, i in enumerate(range(0, len(enodes), 4)):
                n1, n2, n3, n4 = enodes[i:i+4]
                pid = etags[eid]  
                f.write("CQUAD4,{:d},{:d},{:d},{:d},{:d},{:d},0\n".format(
                    etags[eid], pid, n1, n2, n3, n4))
        else:
            raise Exception("Check element type.")

    # elem_id = 1
    # for j in range(Ny):
    #     for i in range(Nx):
    #         n1 = nodes[(i, j)]
    #         n2 = nodes[(i+1, j)]
    #         n3 = nodes[(i+1, j+1)]
    #         n4 = nodes[(i, j+1)]
    #         f.write("CQUAD4,{:d},{:d},{:d},{:d},{:d},{:d},0\n".format(
    #             elem_id, elem_id, n1, n2, n3, n4))
    #         elem_id += 1
    
    # === MAterial ===
    line = "MAT8     1       "
    for val in props:
        if isinstance(val, float) and abs(val) >= 1e6:
            base, exp = "{:.3e}".format(val).split("e")
            base = base.rstrip("0").rstrip(".")
            if "." not in base:
                base += "."  # forza il punto decimale
            exp_sign = "+" if int(exp) >= 0 else "-"
            exp_val = str(int(exp)).lstrip("-+0")
            if exp_val == "":
                exp_val = "0"
            formatted = "{}{}{}".format(base, exp_sign, exp_val)
        else:
            formatted = str(val)
            if "." not in formatted:
                formatted += "."

        line += "{:<8}".format(formatted)

    f.write(line.strip() + "\n")

    # === LOAD CASE ====
    f.write("SPCADD   2       1\n")
    # === Vincoli (SPC) ===
    spc_id = 1
    tol=1e-6
    if side_to_fix  == "left":
        fixed_nodes = [nid for nid, (x, y, z) in nodes.items() if abs(x - 0.0) < tol]
    elif side_to_fix  == "right":
        fixed_nodes = [nid for nid, (x, y, z) in nodes.items() if abs(x - Lx) < tol]
    elif side_to_fix  == "bottom":
        fixed_nodes = [nid for nid, (x, y, z) in nodes.items() if abs(y - 0.0) < tol]
    elif side_to_fix  == "top":
        fixed_nodes = [nid for nid, (x, y, z) in nodes.items() if abs(y - Ly) < tol]
    else:
        fixed_nodes = []

    for nid in fixed_nodes:
        f.write("SPC     {:<8d} {:<8d} 123456\n".format(spc_id, nid))

    # spc_id = 1
    # if side_to_fix == 'left':
    #     fixed_nodes = [nodes[(0, j)] for j in range(Ny+1)]
    # elif side_to_fix == 'right':
    #     fixed_nodes = [nodes[(Nx, j)] for j in range(Ny+1)]
    # elif side_to_fix == 'bottom':
    #     fixed_nodes = [nodes[(i, 0)] for i in range(Nx+1)]
    # elif side_to_fix == 'top':
    #     fixed_nodes = [nodes[(i, Ny)] for i in range(Nx+1)]
    # else:
    #     fixed_nodes = []

    for nid in fixed_nodes:
        f.write("SPC     {:<8d} {:<8d} 123456\n".format(spc_id, nid))

    # AEROELASTICITY

    if sol == 144 or sol == 145:
        f.write("PARAM   AUNITS  1.\n")
        

        x1, y1, z1 = gmsh.model.getValue(0, p1, [])  # Punto 1
        x4, y4, z4 = gmsh.model.getValue(0, p4, [])  # Punto 4
        x2, y2, z2 = gmsh.model.getValue(0, p2, [])  # Punto 2
        x3, y3, z3 = gmsh.model.getValue(0, p3, [])  # Punto 3
        x12 = x2 - x1  # corda alla root
        x43 = x3 - x4

        # aerodynamics
        f.write("AERO,0,1.,{:.4f},{:.4f}\n".format(Lx, rho_f))
        f.write("AEROS,0,0,{:.4f},{:.4f},{:.4f}\n".format(Lx,Ly*2,Lx*Ly))
        f.write("PAERO1  100001\n")
        f.write("CAERO1,100001,100001,0,{:d},{:d},,,1\n".format(Ny,Nx))
        # f.write(",0.,0.,0.,{:.4f},0.,{:.4f},0.,{:.4f}\n".format(Lx,Ly,Lx))          # TODO: adjust coordinates for sweep and TR
        f.write(",{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(x1, y1, z1, x12, x4, y4, z4, x43))
        f.write("SPLINE4,1,100001,1,,1,,IPS,BOTH\n")
        start_id = 100001
        element_ids = list(range(start_id, start_id + Nelems))
        f.write("AELIST,1")
        line = "," + ",".join(str(eid) for eid in element_ids[:7]) + "\n"
        f.write(line)

        for i in range(7, len(element_ids), 8):
            line = "," + ",".join(str(eid) for eid in element_ids[i:i+8]) + "\n"
            f.write(line)
        
        set1_ids = list(range(1, (Nx+1)*(Ny+1) + 1))
        f.write("SET1,1")
        line = "," + ",".join(str(eid) for eid in set1_ids[:7]) + "\n"
        f.write(line)

        for i in range(7, len(set1_ids), 8):
            line = "," + ",".join(str(eid) for eid in set1_ids[i:i+8]) + "\n"
            f.write(line)
    
    if sol == 144:
        f.write("AESTAT  1       ANGLEA\n")
        f.write("AESTAT  2       SIDES\n")
        f.write("AESTAT  3       ROLL\n")
        f.write("AESTAT  4       PITCH\n")
        f.write("AESTAT  5       YAW\n")
        f.write("AESTAT  6       URDD1\n")
        f.write("AESTAT  7       URDD2\n")
        f.write("AESTAT  8       URDD3\n")
        f.write("AESTAT  9       URDD4\n")
        f.write("AESTAT  10      URDD5\n")
        f.write("AESTAT  11      URDD6\n")
        
        f.write("TRIM,1,0.,{:.4f},ANGLEA,{:.7f},SIDES,0.\n".format(q,np.deg2rad(alpha)))
        f.write(",ROLL,0.,PITCH,0.,YAW,0.,URDD1,0.\n")
        f.write(",URDD2,0.,URDD3,0.,URDD4,0.,URDD5,0.\n")
        f.write(",URDD6,0.\n")
        
        
        f.write("DIVERG,50,{:d},0.0\n".format(Nroots))
        f.write("EIGC,50,CLAN,MAX,,,,{:d}\n".format(Nroots))
    
    elif sol == 145:


        for row in MK:
            row_str = ",".join(f"{x:.6f}" for x in row)
            f.write(f"MKAERO2,{row_str}\n")





        f.write("FLFACT  1       1.\n")
        f.write("FLFACT  2       0.0\n")
        
        f.write("FLFACT,3")
        line = "," + ",".join(str(eid) for eid in V_range[:7]) + "\n"
        f.write(line)
        for i in range(7, len(V_range), 8):
            line = "," + ",".join(str(eid) for eid in V_range[i:i+8]) + "\n"
            f.write(line)

        f.write("FLUTTER 1       PK      1       2       3                       .001\n")

    f.write("ENDDATA\n")

gmsh.finalize()
print(f"File scritto: {filename}.bdf")

## RUN NASTRAN

if run:
    print("Running NASTRAN win SOL {:d}".format(sol))
    if os.name == 'nt':
        command = "C:\Program Files\MSC.Software\MSC_Nastran\\2024.1\\bin\\nastran "+filename+".bdf scr=yes old=no"
    elif os.name == 'posix':
            command = "nast20234 "+filename+".bdf scr=yes old=no"

    else:
        print(os.name," platform not recognized")
        
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    print(result.stdout)

## SAVE VTK
if (save_vtk):
    if op2:
        vtk_filename = filename + '.vtu'
        nastran_to_vtk(bdf_filename, op2_filename, vtk_filename)
    else:
        raise "You must activate op2 output"
    
## PLOT RESULTS

if (plot):
    if sol == 103:
        res = utilities.print_vib(filename)
        print(f"{'Modo':>4} | {'lam':>12} | {'rad/s':>12} | {'Hz':>10}")
        print("-" * 47)
        for i, (lam, s, h) in enumerate(res, start=1):
            print(f"{i:4} | {lam:12.4e} | {s:12.4f} | {h:10.4f}")
    elif sol == 144:
        q_dyna = utilities.print_div(filename)
        v = np.sqrt(2*q_dyna/rho_f)
        print("Divergence velocities:\n",v)

        uz_tip_LE, uz_tip_TE = utilities.print_disp_SAA(filename)
        print("uz_tip_LE, uz_tip_TE: ", uz_tip_LE, uz_tip_TE)
        print("twist: ", uz_tip_LE -uz_tip_TE)
    elif (sol == 145):
        utilities.plot_nast(filename)