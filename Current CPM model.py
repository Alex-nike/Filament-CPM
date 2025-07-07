#Final code with working filaments. No active movment so does not move.

import copy
from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import matplotlib.colors as mcolors
from queue import Queue
from collections import deque

#Initialize grid
GRID_SIZE = 75
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

#Define cell type and minumum sizes for connectivity test
CELL_TYPE = [0,1,2,3,4,5] 
min_sizes = {
        CELL_TYPE[0]: 20,  # ECM
        CELL_TYPE[1]: 10,   # Cytoplasm
        CELL_TYPE[2]: 5,   # Nucleus
        CELL_TYPE[3]: 5,   # Inner filament
        CELL_TYPE[4]: 10,   # Outer filament
    }

neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Defines cell type colors
cell_colors = mcolors.ListedColormap([
    'mediumslateblue',    # 0 - empty
    'lightseagreen',    # 1 - cytoplasm
    'gold',     # 2 - nucleus
    'darkorange',      # 3 - inner filament
    'crimson',    # 4 - outer filament
    "black"        #5 - barrier
])
bounds = [0, 1, 2, 3, 4, 5,6] 
norm = mcolors.BoundaryNorm(bounds, cell_colors.N)

#Constants
r1 = 8
r2 = 4

LAMBDA_VOLUME = 1
LAMBDA_VOLUME_NUC = 1
LAMBDA_SURFACE_NUC = 8
LAMBDA_SURFACE = 0.3
LAMBDA_FORCE_NUC = 2
LAMBDA_FORCE = 1
LAMBDA_FIL= 1.5

J_CN = -0.5
J_CM = -0.1
J_NM = 300
J_CC = -1
J_NN = -1

TARGET_VOLUME_NUC = np.pi*r2**2
TARGET_SURFACE = 2*np.pi*r1
TARGET_VOLUME = np.pi*(r1**2-r2**2)
TARGET_NUCLEUS = 2*np.pi*r2

persistance_mu = 10**3

TEMPERATURE = 800
MC_STEPS = 3500
interval = 10

kx = 50

ATTEMPTS_PER_STEP = 1000



#====================================
#List of helper functions
#====================================

def COM(grid,component):
"""
Calculates the COM of a the cell or of the nucleus

Returns com: (x,y)
"""

    #Boolean mask selecting cell or nucleus center of mass
    mask_cell = (grid == CELL_TYPE[1]) | (grid == CELL_TYPE[2]) | (grid == CELL_TYPE[3]) | (grid == CELL_TYPE[4])
    mask_nuc = grid == CELL_TYPE[2]

    com = np.zeros(2)

    if component  == "cell":
        total_mass = np.count_nonzero(mask_cell)
        if total_mass == 0:
            return np.array([0.0, 0.0])
        indices = np.argwhere(mask_cell)
        com = np.mean(indices, axis=0)
    
    elif component == "nuc":
        total_mass = np.count_nonzero(mask_nuc)
        if total_mass == 0:
            return np.array([0.0, 0.0])
        indices = np.argwhere(mask_nuc)
        com = np.mean(indices, axis=0)
    else:
        print("COM not found")
    return com


def max_length(grid):
"""
Calculates the maximum length of the ceell, used for persistance term

Returns max_length
"""
    presence_x = np.any(grid != 0, axis=1)
    presence_y = np.any(grid != 0, axis=0)
    return max(np.sum(presence_x), np.sum(presence_y))



def valid_neighbors(x, y, grid):
    """
    Returns the coordinates of each 4-neighbor as well as their type
    
    Returns neighbors: list of tuples and type_neighbors: a list of each type per coord
    """
    neighbors = []
    type_neighbors = []
    for dx, dy in neighbor_offsets:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
            type_neighbors.append(grid[nx, ny])
    return neighbors, type_neighbors


#Finds the coordinates of the perimeter of the cell
def perimeter(grid, component):
"""
Finds the coordinates of the perimeter of the cell

Returns: out_perimeter_coords and in_perimeter_coords
"""
    out_perimeter_coords = set()
    in_perimeter_coords = set()

    height, width = grid.shape

    for x in range(height):
        for y in range(width):
            if grid[x, y] != component:
                continue

            _, neighbor_types = valid_neighbors(x, y, grid)

            if 0 in neighbor_types:
                out_perimeter_coords.add((x, y))

            if component == 1 and 2 in neighbor_types:
                in_perimeter_coords.add((x, y))

    return out_perimeter_coords, in_perimeter_coords

#Precomuptes the neighbors of each filament
def precompute_fil_neighbors(coords, grid):
"""
Precomuptes the neighbors of each filament

Returns: neighbor_list and Type_list
"""
    neighbor_list = []
    type_list = []
   
    for coord in coords:
        if not (isinstance(coord, (tuple, list)) and len(coord) == 2):
            raise ValueError(
                f"Each entry in `coords` must be a 2‐tuple/list (x, y), but got: {coord}"
            )
        cx, cy = coord

        n_coords, n_types = valid_neighbors(cx, cy, grid)

        neighbor_list.append(set(n_coords))

        type_list.append(n_types.copy())

    return neighbor_list, type_list




def bfs_component_size(grid, start, cell_type):
    """
    Find all connected cells of the same type starting from a given position.

    Returns visted: a list of all cells found in BFS search
    """
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        x, y = queue.popleft()
        for nx, ny in valid_neighbors(x, y, grid)[0]:
            if (nx, ny) not in visited and grid[nx, ny] == cell_type:
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    return visited


def is_connected_component(grid, cell_type, ignore_coord=None):
    """
    Check if ALL components of cell_type meet the minimum size requirement.
    Returns False if any component is too small.
    """
    visited = set()
    
    required_size = min_sizes.get(cell_type, 0)
    if required_size == 0:
        return True 

    #Goes through whole grid, the last check makse sure each component is searched once
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if (grid[i, j] == cell_type and 
                (i, j) != ignore_coord and 
                (i, j) not in visited):
                
                component = bfs_component_size(grid, (i, j), cell_type)
                
                #Adds visited to the larger visited array for each compoent
                visited.update(component)
                
          
                if len(component) < required_size:
                    return False  
    
    return True



def is_attached_to_main_body(grid, coord, main_body_type):
    """
    Check if a coordinate is part of or adjacent to a component of main_body_type 
    that meets the minimum size requirement.

    Returns False if not attached to a componet greater than min size
    """
    x, y = coord
    

    required_size = min_sizes.get(main_body_type, 0)

    # If the coordinate itself is of the main_body_type, check its component
    if grid[coord] == main_body_type:
        component = bfs_component_size(grid, coord, main_body_type)
        return len(component) >= required_size
    
    # Otherwise, check adjacent cells
    neighbor_coords, *_ = valid_neighbors(x, y, grid)
    candidate_neighbors = [n for n in neighbor_coords if grid[n] == main_body_type]
    
   #No main body neighbors, it's not attached. Solves filament islands
    if not candidate_neighbors:
        return False  
    
    
    checked_components = set()
        
    #similar to connected function but just with the coords we care about
    for start in candidate_neighbors:
            
        #Skips checked components
        if start in checked_components:
            continue
            
        connected = bfs_component_size(grid, start, main_body_type)
        
        checked_components.update(connected)
        
        component_size = len(connected) if connected else 0
        if component_size >= required_size:
            return True
    
    return False

def check_filament_connectivity(grid, filament_pos, filament_type, anchor_type):
    """
    Check if a filament at filament_pos is connected to cells of anchor_type
    that form a sufficiently large component.

    Returns False if selected filament is not attached to main body
    """
    visited = set()
    queue = deque([filament_pos])
    visited.add(filament_pos)

    while queue:
        x, y = queue.popleft()
        neighbors, _ = valid_neighbors(x, y, grid)
        
        for nx, ny in neighbors:
            if (nx, ny) not in visited:
                #If the neighbor could be an anchor, check if it is attached to the main body
                if grid[nx, ny] == anchor_type:
                    if is_attached_to_main_body(grid, (nx, ny), anchor_type):
                        return True
                 #otherwise it is a filament which we just skip and check respectively           
                elif grid[nx, ny] == filament_type:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return False


def comprehensive_connectivity_check(grid, x, y, old_type, new_type):
    """Tests if moving new type onto old type maintains conectivity

    Use the prior connectivity functions to test globally if connectibity is broken
    If True, no connectivity is broken
"""
        

    #Creates a temp grip to test connectivity
    original = grid[x, y]
    grid[x, y] = new_type
    
    try:
        #Basic checks for old and new type
        if old_type in min_sizes:
            if not is_connected_component(grid, old_type, ignore_coord=(x, y)):
                return False
        

        if new_type in min_sizes:
            if not is_connected_component(grid, new_type):
                return False
        
        #Filament check
        filament_check_passed = True
        for filament_outer, filament_inner in Filaments:
            # Verify filament positions still contain filament types, keeps the filaments from disappearing
            if grid[filament_outer] != CELL_TYPE[4] or grid[filament_inner] != CELL_TYPE[3]:
                filament_check_passed = False
                break
            
            # Check outer filament connectivity to cytoplasm
            if not check_filament_connectivity(grid, filament_outer, CELL_TYPE[4], CELL_TYPE[1]):
                filament_check_passed = False
                break
            
            # Check inner filament connectivity to nucleus
            if not check_filament_connectivity(grid, filament_inner, CELL_TYPE[3], CELL_TYPE[2]):
                filament_check_passed = False
                break
        
        return filament_check_passed
    
    finally:
        grid[x, y] = original


#=====================
#Intialization of Grid
#=====================
def cell_field(grid):

    center_x = 40
    center_y = 33
    max_width = 12      
    gap_size = 4       

        #Generates a double funnel barrier
    for x in range(20, 61):
        distance_from_center = abs(x - center_x)
        funnel_width = int((distance_from_center / (center_x - 20)) * max_width)

        vertical_half_gap = max(gap_size, funnel_width)

        top_y = center_y + vertical_half_gap
        bottom_y = center_y - vertical_half_gap

        for y in range(grid.shape[1]):
            if y > top_y or y < bottom_y:
                grid[x, y] = CELL_TYPE[5]

    x0, y0 = 10, 33
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x - x0)**2 + (y - y0)**2 <= r2**2:
                grid[x, y] = CELL_TYPE[2]
            elif r2**2 <= ((x - x0)**2 + (y - y0)**2) <= r1**2:
                grid[x,y] = CELL_TYPE[1]
    return x0
            

#Places filaments and keeps track of original length
#Also palces them along roots of unity
def filament_place(grid, n=4):
    out, inner = perimeter(grid, 1)
    filament_list = []

    center = COM(grid,"nuc")  
    inner_set = set(inner)
    outer_set = set(out)

    used_inner = set()
    used_outer = set()

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        direction = direction / np.linalg.norm(direction)

        found_inner = None
        found_outer = None

        #
        for r in np.linspace(1, max(grid.shape), num=500):
            point = tuple(np.round(center + r * direction).astype(int))

            if not (0 <= point[0] < grid.shape[0] and 0 <= point[1] < grid.shape[1]):
                break  

            if found_inner is None and point in inner_set and point not in used_inner and grid[point] != CELL_TYPE[3]:
                found_inner = point
                used_inner.add(point)
                continue

            if found_inner is not None and point in outer_set and point not in used_outer and grid[point] != CELL_TYPE[4]:
                found_outer = point
                used_outer.add(point)
                break

        if found_inner and found_outer:
            grid[found_inner] = CELL_TYPE[3]
            grid[found_outer] = CELL_TYPE[4]
            filament_list.append((found_outer, found_inner))
        else:
            print(f"Could not find valid inner/outer points for angle {np.degrees(angle):.2f}°")

    # Calculate initial lengths
    length_0_list = np.array([
        np.linalg.norm(np.array(fil[0]) - np.array(fil[1]))
        for fil in filament_list
    ])

    return filament_list, length_0_list

x0 = cell_field(grid)

Filaments, length_0 = filament_place(grid)



#==============================
#Hamiltonian terms
#==============================

def volume_energy(grid):
    volume = np.count_nonzero(grid == 1)
    return LAMBDA_VOLUME * (volume - TARGET_VOLUME) ** 2

def volume_energy_nuc(grid):
    volume = np.count_nonzero(grid == 2)
    return LAMBDA_VOLUME_NUC * (volume - TARGET_VOLUME_NUC) ** 2

def surface_energy(grid):
    surface = 0
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == CELL_TYPE[1]:
                neighbors,_ = valid_neighbors(x, y,grid)
                for nx, ny in neighbors:
                    if grid[nx, ny] != CELL_TYPE[1]:
                        surface += 1
    return LAMBDA_SURFACE * (surface - TARGET_SURFACE)**2

def surface_energy_nuc(grid):
    surface = 0
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == CELL_TYPE[2]:
                neighbors,_ = valid_neighbors(x, y,grid)
                for nx, ny in neighbors: 
                    if grid[nx, ny] != CELL_TYPE[2]:
                        surface += 1
    return LAMBDA_SURFACE_NUC * (surface - TARGET_NUCLEUS)**2

def adhesion_energy_ext(grid):
    adhesion_energy_sum = 0
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == CELL_TYPE[1]:
                neighbors,_ = valid_neighbors(x, y,grid)
                for nx, ny in neighbors:
                    if grid[nx, ny] == CELL_TYPE[0]:
                        adhesion_energy_sum += J_CM
                    elif grid[nx,ny] == CELL_TYPE[1]:
                        adhesion_energy_sum += J_CC
                    elif grid[nx,ny] == CELL_TYPE[2]:
                        adhesion_energy_sum += J_CN
    return adhesion_energy_sum

def adhesion_energy_int(grid):
    adhesion_energy_sum = 0
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == CELL_TYPE[2]:
                neighbors,_ = valid_neighbors(x, y,grid)
                for nx, ny in neighbors:
                    #check for each intermedium interaction
                    if grid[nx, ny] == CELL_TYPE[0]:
                        adhesion_energy_sum += J_NM
                    elif grid[nx,ny] == CELL_TYPE[2]:
                        adhesion_energy_sum += J_NN
    return adhesion_energy_sum

#Hooke's law
def extend_force(extension):
    return -LAMBDA_FORCE*extension

#Determines the total filament work on the nucleus and between the two ends of the filament
def filament_work(grid):
    COM_nuc = COM(grid, "nuc")

    # Computes the current length of filaments
    fil_length = np.array([
        np.linalg.norm(np.array(fil[0]) - np.array(fil[1]))
        for fil in Filaments
    ])

    if len(fil_length) == 0:
        return 0.0

    # Extension of the filaments w.r.t original length and force calulated
    extension = fil_length - length_0
    force = extend_force(extension) 

    Total_work= 0.0

    #For each filament, the displacement between the inner filament point and the COM of the nucleus is calculated as well as between that and the end point.
    #The total work is then the dot product of the force and displacement from the original length i.e. the extension. The forces act along the displacement vector though.
    for idx, fil in enumerate(Filaments):
        f = force[idx]

        endpoint_outer = np.array(fil[0])
        endpoint_inner = np.array(fil[1])


        disp_inner = endpoint_inner - COM_nuc
        disp_fil = endpoint_outer - endpoint_inner

        norm_inner = np.linalg.norm(disp_inner)
        norm_fil = np.linalg.norm(disp_fil)

        unit_inner = disp_inner / norm_inner
        unit_fil = disp_fil / norm_fil


        extension_inner_vec = unit_inner * (norm_inner - r2)
        extension_fil_vec = unit_fil * (norm_fil - length_0[idx])


        force_nuc = LAMBDA_FORCE_NUC * extension_inner_vec
        force_fil = LAMBDA_FIL * extension_fil_vec

        work_nuc = np.dot(force_nuc, unit_inner)
        work_fil = np.dot(force_fil, unit_fil)

        Total_work += f * (work_nuc + work_fil)

    return float(Total_work)

#gradient function
def gradient(point):
    x, y = point[0], point[1]
    target_x= 60  

    if x < 60:
        return kx * (x - target_x)**2 

 
    

def total_energy_cell(grid):
    com = COM(grid,"cell")
    return volume_energy(grid) + surface_energy(grid) +  surface_energy_nuc(grid) + adhesion_energy_ext(grid) + adhesion_energy_int(grid) + volume_energy_nuc(grid)  + gradient(com)


# ====================
# Filament Logic
# ====================

#Simple swap logic
def simple_swap(grid, Filaments, old_coord, neighbor_coord, filament_type, perim, perim_type):

"""
Checks if you can simple swap the old type and new type
Returns True if swapped
"""
    x, y = old_coord
    nx, ny = neighbor_coord

    swapped = False

    # Check if both points on perimeter
    if not (((nx, ny) in perim) and ((x, y) in perim)):
        return swapped

    # Identify which filament is to be swapped
    fil_index = None
    for i, (outer, inner) in enumerate(Filaments):
        if filament_type == CELL_TYPE[3] and inner == (x, y):
            fil_index = i
            break
        elif filament_type == CELL_TYPE[4] and outer == (x, y):
            fil_index = i
            break
        elif filament_type == CELL_TYPE[3] and inner == (nx, ny):
            fil_index = i
            break
        elif filament_type == CELL_TYPE[4] and outer == (nx, ny):
            fil_index = i
            break

    if fil_index is None:
        return swapped


    # Swap in the grid
    grid[x, y], grid[nx, ny] = grid[nx, ny], grid[x, y]

    # Check connectivity
    if not check_filament_connectivity(grid, (nx, ny), filament_type, perim_type):
        grid[x, y], grid[nx, ny] = grid[nx, ny], grid[x, y]  # revert
        return swapped

    #Updates filament list
    outer, inner = Filaments[fil_index]
    if filament_type == CELL_TYPE[3]:  
        if inner == (x, y):
            Filaments[fil_index] = (outer, (nx, ny))
        elif inner == (nx, ny):
            Filaments[fil_index] = (outer, (x, y))
    else:  
        if outer == (x, y):
            Filaments[fil_index] = ((nx, ny), inner)
        elif outer == (nx, ny):
            Filaments[fil_index] = ((x, y), inner)

    swapped = True  
    return swapped

def passive_movement(old_coord, new_type, filament_type, neighbors, outer_perim, inner_perim):
""" Checks if any passive movement occurs do to copy attempt
Returns True if passive movement occurs
"""
        
    x, y = old_coord
    moved = False

    #For generality, defines extend and retract type as well as which perimeter is used per filament type
    if filament_type == CELL_TYPE[4]:  
        extend_type = CELL_TYPE[1]     
        retract_type = CELL_TYPE[0]    
        valid_perimeter = outer_perim
    elif filament_type == CELL_TYPE[3]:  
        extend_type = CELL_TYPE[2]       
        retract_type = CELL_TYPE[1]      
        valid_perimeter = inner_perim
    else:
        return moved

    #Initialise filament index and postion for Filaments update
    fil_index = None
    filament_endpoint_pos = None
    
    for idx, (outer, inner) in enumerate(Filaments):
        endpoint = outer if filament_type == CELL_TYPE[4] else inner
        
        # Check if the filament endpoint is adjacent to the changing position and already on the perimeter
        if endpoint in neighbors and endpoint in valid_perimeter:
            filament_endpoint_pos = endpoint
            fil_index = idx
            break

   #If no filament is found, exit
    if fil_index is None or filament_endpoint_pos is None:
        return moved

    # Only proceed if new_type is valid for extension/retraction, failsafe
    if new_type not in (extend_type, retract_type):
        return moved

    #We use a test grid for this
    old_grid = grid.copy()
    old_filaments = copy.deepcopy(Filaments)

    #Try copy attempt
    grid[x,y] = new_type

    # Check what surrounds the current filament endpoint (excluding the position that just changed)
    fil_neighbors = []
    for nx, ny in valid_neighbors(*filament_endpoint_pos, grid)[0]:
            fil_neighbors.append(grid[nx, ny])
    
    fil_neigh_set = set(fil_neighbors)

    # --- Extension: Filament grows into the changing position ---
    if new_type == extend_type and fil_neigh_set.issubset({extend_type}):
        # Move filament endpoint to (x, y) and fill old position with extend_type
        grid[x, y] = filament_type
        grid[filament_endpoint_pos] = extend_type
        
        # Check connectivity after the move
        if check_filament_connectivity(grid, (x, y), filament_type, extend_type):
            outer, inner = Filaments[fil_index]
            if filament_type == CELL_TYPE[4]:
                Filaments[fil_index] = ((x, y), inner)
            else:
                Filaments[fil_index] = (outer, (x, y))
            moved = True
        else:
            # Revert changes
            grid[:, :] = old_grid
            Filaments[:] = old_filaments

    # --- Retraction: Filament pulls back from changing position ---
    elif new_type == retract_type and fil_neigh_set.issubset({retract_type}):
        # Move filament endpoint to (x, y) and fill old position with extend_type
        grid[x, y] = filament_type
        grid[filament_endpoint_pos] = retract_type
        
        # Check connectivity after the move
        if check_filament_connectivity(grid, (x, y), filament_type, retract_type):
            outer, inner = Filaments[fil_index]
            if filament_type == CELL_TYPE[4]:
                Filaments[fil_index] = ((x, y), inner)
            else:
                Filaments[fil_index] = (outer, (x, y))
            moved = True
        else:
            # Revert changes
            grid[:, :] = old_grid
            Filaments[:] = old_filaments

    return moved



#===================
#Flip logic
#===================

def attempt_flip(grid, persist,length):
    #Initialise flags for the swap attempt
    delta_energy = None

    x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)

    old_type = grid[x, y]

    neighbors, _ = valid_neighbors(x, y, grid)

    # Checks if no neighbors
    if not neighbors:
        return delta_energy

    nx, ny = random.choice(neighbors)
    new_type = grid[nx, ny]

    # No attempts can occur if a filament is selected for copy or same-type copy attempted
    if old_type == new_type or CELL_TYPE[5] in (old_type,new_type) or old_type in (CELL_TYPE[3],CELL_TYPE[4]):
        return delta_energy


    # Precompute perimeter and filament data
    outer_perim, inner_perim = perimeter(grid, 1)


    # Initial energy calculation
    old_energy = total_energy_cell(grid) + persist

    #Save the original state of the grid and filaments
    old_grid = grid.copy()
    old_filament = copy.deepcopy(Filaments)


    #Movememnt type flags
    swapped = False
    simple_swapped = False


    #Filament logic check
    for filament_type, perim, perim_type in [
        (CELL_TYPE[4], outer_perim, CELL_TYPE[1]),  # Outer filament
        (CELL_TYPE[3], inner_perim, CELL_TYPE[2])   # Inner filament
        ]:
        # Attempt simple swap
        simple_swapped= simple_swap(grid,Filaments,(x, y), (nx, ny), filament_type, perim, perim_type)
        if simple_swapped:
            swapped = True
            break

        # Attempt passive movement
        swapped = passive_movement((x, y), new_type, filament_type, neighbors,outer_perim,inner_perim)
        if swapped:
            break

    

    # ===========================
    # Non-filament attempt
    # ===========================

    #If no filament logic has occured
    if not swapped and not simple_swapped:
        #Use dull connectivity check, then make the change
        if not comprehensive_connectivity_check(grid, x, y, old_type, new_type):
            return delta_energy
        grid[x, y] = new_type
    

    new_energy = total_energy_cell(grid) + persist
    delta_energy = new_energy - old_energy + filament_work(grid)

    # Metropolis criterion
    if delta_energy <= 0 or random.random() < np.exp(-delta_energy / TEMPERATURE):
        return delta_energy
    else:
        # Revert flip
        if swapped or simple_swapped:
            grid[:, :] = old_grid
            Filaments[:] = old_filament
        else:
            grid[:, :] = old_grid
        return delta_energy
    


#===========================
#Swap attempt, makes the filaments more dynamic around the perimeter
#===========================
def attempt_swap(grid, failed_delta_energy, length):
    """
    Attempt filament swap only when main flip fails, using pre-calculated energy.
    """

    if not Filaments or failed_delta_energy is None:
        return 0.0

    # Make a copy of the grid for reversion
    temp_grid = grid.copy()


    # Pick random filament and endpoint type
    fil_idx = random.randint(0, len(Filaments) - 1)
    endpoint_type = random.randint(0, 1)
    fil_point = Filaments[fil_idx][endpoint_type]

    # Get valid perimeter neighbors
    outer_perim, inner_perim = perimeter(grid, 1)
    perim = outer_perim if endpoint_type == 0 else inner_perim
    neighbors, _ = valid_neighbors(*fil_point, grid)
    valid_neighbors_on_perim = [n for n in neighbors if n in perim]

    if not valid_neighbors_on_perim:
        return 0.0

    neighbor = random.choice(valid_neighbors_on_perim)

    # Attempt the swap in the temp grid
    temp_grid[fil_point], temp_grid[neighbor] = temp_grid[neighbor], temp_grid[fil_point]


    # Evaluate energy
    delta_energy = failed_delta_energy + filament_work(temp_grid)

    if delta_energy <= 0 or random.random() < np.exp(-delta_energy / TEMPERATURE):
        # Accept: commit the temp grid to main grid
        grid[:, :] = temp_grid

        # Update filament tracking
        if endpoint_type == 0:
            Filaments[fil_idx] = (neighbor, Filaments[fil_idx][1])
        else:
            Filaments[fil_idx] = (Filaments[fil_idx][0], neighbor)

        return delta_energy
    else:
        # Reject: do nothing, original grid unchanged
        return 0.0

        



#======================
#Visualization and main simulation function
#======================

#This array is used to track the COM of cell on the video
com_history = []
com_nucl_list = []

def visualize(grid, step, COM,com_history,COM_nuc,com_nucl_list):
    #Fixes difference betwene numpy grid and python grid
    grid = np.transpose(grid)
        
    #Adds new COM into the array
    com_history.append(COM)
    com_nucl_list.append(COM_nuc)
    
    # Unpacks COM coordinates
    xs, ys = zip(*com_history)
    xn,yn = zip(*com_nucl_list)

    # Plot COM history as a red line, tracking cell movement
    plt.plot(xs, ys, 'r-', linewidth=1,label = "Cell COM")   
    plt.plot(xs[-1], ys[-1], 'ro',markersize=1) 
        
    #Plot COM nuc hist as blue line
    plt.plot(xn, yn, 'b-', linewidth=1,label = "Nucleus COM")   
    plt.plot(xn[-1], yn[-1], 'bo',markersize=1)
   

 #Draws a quadratic bezier curve between filament points, if there's an error just make a straight line
    for outer, inner in Filaments:
        x0, y0 = outer
        x1, y1 = inner

        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.5  

        try:
            tck, _ = splprep([[x0, mx, x1], [y0, my, y1]], k=2)
            u_fine = np.linspace(0, 1, 100)
            smoothed = splev(u_fine, tck)
            plt.plot(smoothed[0], smoothed[1], color='black', linewidth=0.5, alpha=0.5)
        except ValueError:
            plt.plot([x0, x1], [y0, y1], color='black', linewidth=0.5, alpha=0.5) 
    
    plt.imshow(grid, cmap = cell_colors, norm=norm, origin= "lower")
    plt.title(f'Step {step}')
    plt.colorbar(ticks=[0, 1, 2, 3, 4, 5], label='Cell Types')
    plt.legend(loc='upper right')

    output_dir = 'Cell_simulation_img'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"step_{step:04d}.png")
    plt.savefig(output_path)
    plt.close()


def run_simulation():
    total_energy_list = []
    last_com = None
    global TEMPERATURE
    global J_NM

    #initialise the velocity and COM as queues with 0,0 for velocity and 0,[x0,y0]
    velocity = Queue(maxsize=2)
    COM_dyn = Queue(maxsize=2)

    velocity.put(np.array([[0,0]]))
    velocity.put(np.array([[1000,0]]))

    COM_dyn.put(np.array([0,0]))
    COM_dyn.put(COM(grid, "cell"))


    
    for step in range(MC_STEPS):

        #Persistance term based on flip attempt so within each step loop, we calculate the new persistnace term
        #Adapts the persistnace lagrange multiplier term from M. Scianna, L. Preziosi (2013) into a dynamic function
        #Queue actions are taken at the end of each flip, even if no change occurs, effectively setting v = 0 that step.
        l = max_length(grid)
        mu_pers = persistance_mu*(l/(2*r2) - 1)
        delta_v = velocity.queue[1]-velocity.queue[0]
        persist = mu_pers*np.linalg.norm(delta_v)**2



        # Perform multiple attempts per MCS
        for _ in range(ATTEMPTS_PER_STEP):   
            energy = attempt_flip(grid,persist,l)
            attempt_swap(grid,energy,l)


        COM_dyn.get()
        COM_dyn.put(COM(grid, 'cell'))
        velocity.get()
        velocity.put(COM_dyn.queue[1]-COM_dyn.queue[0])


        current_energy = total_energy_cell(grid) + persist
        total_energy_list.append(current_energy)

        

        if step % interval == 0:
            visualize(grid, step,COM(grid,"cell") ,com_history,COM(grid,"nuc"),com_nucl_list)
            print(f'Step {step}: Energy = {current_energy}')

    com_cell,com_nuc = COM(grid, "cell"), COM(grid,"nuc")

    visualize(grid, MC_STEPS,com_cell,com_history,com_nuc,com_nucl_list)
    print(f"The displacement of the cell COM in the x direction is  {com_cell[0]-x0}")
    print(f"The displacement of the nucleus COM in the x direction is {com_nuc[0]-x0}")

    

if __name__ == "__main__":
    run_simulation()
