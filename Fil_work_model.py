#WIP build

from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import matplotlib.colors as mcolors
from queue import Queue
from collections import Counter



#Initialize grid
GRID_SIZE = 75
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

#Define cell type and compartment: CELL_TYPE defines medium, cytoplasm and nuclues 
CELL_TYPE = [0,1,2,3,4] 

# Defines cell type colors
cell_colors = mcolors.ListedColormap([
    'indigo',    # 0 - empty
    'lightseagreen',    # 1 - cytoplasm
    'gold',     # 2 - nucleus
    'darkorange',      # 3 - inner filament
    'maroon'    # 4 - outer filament
])

#Custom color norm
bounds = [0, 1, 2, 3, 4, 5] 
norm = mcolors.BoundaryNorm(bounds, cell_colors.N)

#Constants
r1 = 8
r2 = 4
LAMBDA_VOLUME = 1
LAMBDA_VOLUME_NUC = 1
LAMBDA_SURFACE_NUC = 1
LAMBDA_SURFACE = 0.5
LAMBDA_FORCE_NUC = 1
LAMBDA_FORCE = 1
J_CN = 1
J_CM = 0
J_NM = 1000
J_CC = -0.5
J_NN = -10
TARGET_VOLUME_NUC = np.pi*r2**2
TARGET_SURFACE = 2*np.pi*r1
TARGET_VOLUME = np.pi*(r1**2-r2**2)
TARGET_NUCLEUS = 2*np.pi*r2
persistance_mu = 10**5
TEMPERATURE = 100
MC_STEPS = 3500
interval = 10

# Number of pixel-flip attempts per MC step
ATTEMPTS_PER_STEP = 1000


#====================================
#List of Pre-built Functions used in Logic
#====================================

#Here we define the set class for the Union Find algorithm. This class uses sets and unions instead of BFS and DFS so test connectedness.
#This is typically faster than BFS or DFS in larger grids like this one
class UnionFind:
    def __init__(self, size):
        # Initialize each element as a tree with only it as its root
        self.parent = list(range(size))

    # Finds the root of x with path compression
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    # Joins the sets containing x and y
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x  # or vice versa


    # Returns the number of distinct connected components
    def groups(self):
        return len(set(self.find(x) for x in range(len(self.parent))))



#Calculates the COM of a certain type of cell component
def COM(grid,type1,type2 = -1):
    total_mass = 0
    x_sum = 0
    y_sum = 0

    #Uses each area of cell to calculate COM with the x_sum and Y_sum weights being the coords
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if type2 != -1:
                if grid[x, y] == type1 or grid[x, y] == type2 or grid[x,y] == CELL_TYPE[3] or grid[x,y] == CELL_TYPE[4]:
                    total_mass += 1
                    x_sum += x
                    y_sum += y
            else:
                if grid[x, y] == type1:
                    total_mass += 1
                    x_sum += x
                    y_sum += y

    if total_mass == 0:
        return np.array([0.0, 0.0])  

    x_mean = x_sum / total_mass
    y_mean = y_sum / total_mass
    
    return np.array([x_mean,y_mean])


#Helps implement persistance term from M. Scianna, L. Preziosi (2013)
def length(grid):
    """Find the largest length of the cell"""
    grid_x = np.zeros(GRID_SIZE)
    grid_y = np.zeros(GRID_SIZE)

    #These two operations reduce the x,y profile into 1-dimension
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] != 0:
                grid_y[y] += 1
                grid_x[x] += 1

    max_l_x = max(grid_x)
    max_l_y = max(grid_y)

    return max([max_l_x,max_l_y])


#Returns 4 valid neighbors and their types at a point
def valid_neighbors(x, y, grid):
    neighbors = []
    type_neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
            type_neighbors.append(grid[nx, ny])
    return neighbors, type_neighbors


#Find the coordinates of the perimeter of the cell
def perimeter(grid, component):
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

def precompute_fil_neighbors(coords, grid):
    """
    Given `coords` = a list/array of (x, y) tuples (filament tips),
    returns:
      - neighbor_list[i] = set of valid neighbor‐coordinates around coords[i]
      - type_list[i]     = list of the corresponding grid‐values for those neighbors

    Example usage:
        outer_coords = [tip for (tip, _) in Filaments]
        neighbors_outer, types_outer = precompute_fil_neighbors(outer_coords, grid)
    """
    neighbor_list = []
    type_list = []

    for coord in coords:
        # Ensure coord is a length‐2 sequence before unpacking
        if not (isinstance(coord, (tuple, list)) and len(coord) == 2):
            raise ValueError(
                f"Each entry in `coords` must be a 2‐tuple/list (x, y), but got: {coord}"
            )
        cx, cy = coord

        # Now get the valid neighbors of (cx, cy)
        n_coords, n_types = valid_neighbors(cx, cy, grid)

        # Store a set of the neighbor‐coordinates for fast membership tests
        neighbor_list.append(set(n_coords))

        # Store the corresponding neighbor types in the same order as n_coords
        type_list.append(n_types.copy())

    return neighbor_list, type_list



#Checks that cell components are sufficiently contiguous using Union Find based on a minimum size
def is_connected_component(grid, cell_type, min_size, ignore_coord=None):
    coords = []

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == cell_type and (i, j) != ignore_coord:
                coords.append((i, j))

    if not coords:
        return False

    coord_to_index = {coord: idx for idx, coord in enumerate(coords)}
    uf = UnionFind(len(coords))

    for idx, (x, y) in enumerate(coords):
        for nx, ny in valid_neighbors(x, y, grid)[0]:
            if (nx, ny) in coord_to_index:
                uf.union(idx, coord_to_index[(nx, ny)])

    root_counts = Counter(uf.find(idx) for idx in range(len(coords)))
    max_group_size = max(root_counts.values(), default=0)

    return max_group_size >= min_size  

#This makes sure that the filaments can move but only if attached to their respective cell_type
def is_attached_to_main_body(grid, coord, cell_type, min_size=10):

    neighbor_coords, _ = valid_neighbors(*coord, grid)
    candidate_neighbors = [n for n in neighbor_coords if grid[n] == cell_type]

    # If no adjacent pixels are of the correct type, skip early
    if not candidate_neighbors:
        return False

    #Temporarely removes the neighbors of the proposed cell type
    for n in candidate_neighbors:
        grid[n] = -1  # Temporarily mark

    #If there is still a main body, return true but if the filament is floating with an island of cytoplasm or nucleus this returns False
    attached = is_connected_component(grid, cell_type, min_size)

    #Reverts grid
    for n in candidate_neighbors:
        grid[n] = cell_type

    return attached

#To deal with the edge case where two filaments are next to eachother, these help functions are used
def are_filaments_adjacent(fil1_outer, fil1_inner, fil2_outer, fil2_inner, max_distance=2):
    """Check if two filaments are too close to each other"""
    # Check all endpoint combinations
    distances = [
        np.linalg.norm(np.array(fil1_outer) - np.array(fil2_outer)),
        np.linalg.norm(np.array(fil1_outer) - np.array(fil2_inner)),
        np.linalg.norm(np.array(fil1_inner) - np.array(fil2_outer)),
        np.linalg.norm(np.array(fil1_inner) - np.array(fil2_inner))
    ]
    return min(distances) <= max_distance

#Checks whether filaments are adjacent
def check_filament_spacing(grid, filament_idx, new_outer_pos=None, new_inner_pos=None):
    """Ensure filament doesn't get too close to other filaments"""
    if len(Filaments) <= 1:
        return True
    
    # Get current or proposed positions
    outer_pos = new_outer_pos if new_outer_pos else Filaments[filament_idx][0]
    inner_pos = new_inner_pos if new_inner_pos else Filaments[filament_idx][1]
    
    # Check against all other filaments
    for idx, (other_outer, other_inner) in enumerate(Filaments):
        if idx == filament_idx:
            continue
            
        if are_filaments_adjacent(outer_pos, inner_pos, other_outer, other_inner):
            return False
    
    return True

#This connectedness test includes the filament edge case
def enhanced_attachment_check(grid, pos, cell_type, filament_idx=None):
    """Attachment check that accounts for nearby filaments"""
    # Basic attachment check
    if cell_type == CELL_TYPE[1]:
         if not is_attached_to_main_body(grid, pos, cell_type, min_size=10):
            return False
    elif cell_type == CELL_TYPE[1]:
        if not is_attached_to_main_body(grid, pos, cell_type, min_size=30):
            return False
    
    # If this is a filament position, check for interference with other filaments
    if filament_idx is not None:
        # Counts how many filament endpoints are nearby
        nearby_filament_count = 0
        for idx, (outer, inner) in enumerate(Filaments):
            if idx == filament_idx:
                continue
            
            outer_dist = np.linalg.norm(np.array(pos) - np.array(outer))
            inner_dist = np.linalg.norm(np.array(pos) - np.array(inner))
            
            #If filaments are 2 or less neighbors away
            if outer_dist <= 1 or inner_dist <= 1:  
                nearby_filament_count += 1
        
        # Be more lenient with attachment if filaments are crowded
        if nearby_filament_count >= 2:
            return True  # Skip strict attachment check when crowded
    
    return True

def extend_force(extension):
    return LAMBDA_FORCE*extension**2



#=====================
#Intialization of Grid
#=====================

def cell_field(grid):
    x0, y0 = 10,33
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x - x0)**2 + (y - y0)**2 <= r2**2:
                grid[x, y] = CELL_TYPE[2]
            elif r2**2 <= ((x - x0)**2 + (y - y0)**2) <= r1**2:
                grid[x,y] = CELL_TYPE[1]

    return x0

#Places filaments and keeps track of original length
def filament_place(grid):
    out, inner = perimeter(grid, 1)
    filament_list = []

    # Convert sets to lists so we can use random.choice
    inner_list = list(inner)
    outer_list = list(out)


    used_inner = set()
    used_outer = set()

    for _ in range(4):
        # Get a unique inner  and outer point
        inner_candidates = [pt for pt in inner_list if pt not in used_inner and grid[pt] != CELL_TYPE[3]]
        outer_candidates = [pt for pt in outer_list if pt not in used_outer and grid[pt] != CELL_TYPE[4]]

        if not inner_candidates or not outer_candidates:
            print("Ran out of valid inner or outer points")
            break

        innerpoint = random.choice(inner_candidates)
        outerpoint = random.choice(outer_candidates)

        grid[innerpoint] = CELL_TYPE[3]
        grid[outerpoint] = CELL_TYPE[4]

        filament_list.append((outerpoint, innerpoint))
        used_inner.add(innerpoint)
        used_outer.add(outerpoint)

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
    volume = np.sum(grid == 1)
    return LAMBDA_VOLUME * (volume - TARGET_VOLUME)**2

def volume_energy_nuc(grid):
    volume = np.sum(grid == 2)
    return LAMBDA_VOLUME_NUC * (volume - TARGET_VOLUME_NUC)**2

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
                    elif grid[nx,ny] == CELL_TYPE[1]:
                        adhesion_energy_sum += J_CN
                    elif grid[nx,ny] == CELL_TYPE[2]:
                        adhesion_energy_sum += J_NN
    return adhesion_energy_sum

#Determines the filament force on the nucleus and onto the filaments themselves so they move together
#Only uses the strongest filament force
def filament_force_bidirect(grid):
    COM_nuc = COM(grid, CELL_TYPE[2])
    
    fil_length = np.array([
        np.linalg.norm(np.array(fil[0]) - np.array(fil[1]))
        for fil in Filaments
    ])
    
    if len(fil_length) == 0:
        return 0.0
    
    extension = fil_length - length_0
    force = extend_force(extension)
    
    # Find filament with largest force
    max_force_idx = np.argmax(abs(force))
    max_outer = Filaments[max_force_idx][0]
    max_inner = Filaments[max_force_idx][1]
    
    # Force on outer endpoint is based on nucleus
    outer_sep_dist = (np.sum(np.array(max_outer)**2) + 
                     np.sum(COM_nuc**2) - 
                     2*np.dot(max_outer, COM_nuc))
    
    # Force on inner endpoint is based on outer and inner
    inner_sep_dist = (np.sum(np.array(max_inner)**2) + 
                     np.sum(np.array(max_outer)**2) - 
                     2*np.dot(max_inner, max_outer))
    
    return LAMBDA_FORCE_NUC * (outer_sep_dist + inner_sep_dist)


def gradient(point):
    x, y = point[0], point[1]
    target_x= 60  # Exit of the channel

    # Define regions
    if x < 60:
        # Region before tunnel exit: strong pull toward the exit point
        kx = 10
        return kx * (x - target_x)**2 

 
    

def total_energy_cell(grid):
    com = COM(grid,1,2)
    return volume_energy(grid) + surface_energy(grid) +  surface_energy_nuc(grid) + adhesion_energy_ext(grid) + adhesion_energy_int(grid) + volume_energy_nuc(grid)  + gradient(com)




#===================
#Flip logic
#===================

def attempt_flip(grid, persist):
    x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
    old_type = grid[x, y]

    neighbors, _ = valid_neighbors(x, y, grid)

    # Checks if no neighbors
    if not neighbors:
        return 0.0

    nx, ny = random.choice(neighbors)
    new_type = grid[nx, ny]

    # No attempts can occur if a filament is selected or same-type copy attempted
    if new_type in (CELL_TYPE[3], CELL_TYPE[4]) or old_type == new_type:
        return 0.0

    # Precompute perimeter and filament data
    outer_perim, inner_perim = perimeter(grid, 1)
    outer_coords = [coord[0] for coord in Filaments]
    inner_coords = [coord[1] for coord in Filaments]
    neighbors_outer, _ = precompute_fil_neighbors(outer_coords, grid)
    neighbors_inner, _ = precompute_fil_neighbors(inner_coords, grid)

    # Initial energy calculation
    old_energy = total_energy_cell(grid) + persist

    # Flags for filament management
    swapped = False
    simple_swap = False
    initial_out_pos = None
    initial_in_pos = None
    filament_idx = None

    # ===========================
    # Outer filament logic Integrin
    # ===========================
    for idx, neighbor_set in enumerate(neighbors_outer):
        #Check if x,y is a neighbor of the outer filament
        if (x, y) in neighbor_set:
            initial_out_pos = Filaments[idx][0]
            filament_idx = idx

            # Check if filaments would be adjacent, if so do not continue
            proposed_outer_pos = (nx, ny) if (nx, ny) == initial_out_pos else (x, y)
            if not check_filament_spacing(grid, idx, new_outer_pos=proposed_outer_pos):
                continue  

            # Simple inner perimeter swap
            if (nx, ny) == initial_out_pos and (x, y) in outer_perim:
                grid[x, y], grid[nx, ny] = grid[nx, ny], grid[x, y]
                Filaments[idx] = ((x, y), Filaments[idx][1])  
                simple_swap = True
                break

            # Store original state before test move
            original_grid_xy = grid[x, y]
            original_grid_initial = grid[initial_out_pos]
            
            #Test move
            grid[x, y] = new_type

            outer_perim,_ = perimeter(grid, 1)
            if (x, y) not in outer_perim:
                grid[x, y] = original_grid_xy
                break

            #Find filament neighbors after that move
            _, neighbor_types = valid_neighbors(*initial_out_pos, grid)
            neighbor_types = set(neighbor_types)

            #If filament surrounded by cytoplasm
            if neighbor_types.issubset({CELL_TYPE[1]}):
                #extension into ECM
                grid[x, y] = CELL_TYPE[4]
                grid[initial_out_pos] = CELL_TYPE[1]
                Filaments[idx] = ((x, y), Filaments[idx][1])
                swapped = True
                #Check cytoplasm attachment
                if not enhanced_attachment_check(grid, (x, y), CELL_TYPE[1], filament_idx=idx):
                    # Revert all changes
                    grid[x, y] = original_grid_xy
                    grid[initial_out_pos] = original_grid_initial
                    Filaments[idx] = (initial_out_pos, Filaments[idx][1])
                    return 0.0
            #If surrounded by ECM
            elif neighbor_types.issubset({CELL_TYPE[0]}):
                #Retraction into cytoplasm
                grid[x, y] = CELL_TYPE[4]  
                grid[initial_out_pos] = CELL_TYPE[1]  
                Filaments[idx] = ((x, y), Filaments[idx][1])
                swapped = True
                #Check cytoplasm attachment 
                if enhanced_attachment_check(grid, (x, y), CELL_TYPE[1], filament_idx=idx):
                    # Revert all changes
                    grid[x, y] = original_grid_xy
                    grid[initial_out_pos] = original_grid_initial
                    Filaments[idx] = (initial_out_pos, Filaments[idx][1])
                    return 0.0
            else:
                # Revert test move if conditions not met
                grid[x, y] = original_grid_xy
                break


    # ===========================
    # Inner filament logic
    # ===========================
    if not swapped:
        for idx, neighbor_set in enumerate(neighbors_inner):
            #Same logic for inner filament
            if (x, y) in neighbor_set:
                initial_in_pos = Filaments[idx][1]
                filament_idx = idx

                # Check adjacency for inner filament
                proposed_inner_pos = (nx, ny) if (nx, ny) == initial_in_pos else (x, y)
                if not check_filament_spacing(grid, idx, new_inner_pos=proposed_inner_pos):
                    continue

                # Simple inner perimeter swap
                if (nx, ny) == initial_in_pos and (x, y) in inner_perim:
                    grid[x, y], grid[nx, ny] = grid[nx, ny], grid[x, y]
                    Filaments[idx] = (Filaments[idx][0], (x, y))  # Update inner position to new location
                    simple_swap = True
                    break

                # Store original state before test move
                original_grid_xy = grid[x, y]
                original_grid_initial = grid[initial_in_pos]

                # Attempt filament movement and find neighbors
                grid[x, y] = new_type

                #keeps old_type on inner perim
                _, inner_perim = perimeter(grid, 1)
                if (x, y) not in inner_perim:
                    grid[x, y] = original_grid_xy
                    break
                _, neighbor_types = valid_neighbors(*initial_in_pos, grid)
                neighbor_types = set(neighbor_types)

                #If filament surrounded by cytoplasm or ECM
                if neighbor_types.issubset({CELL_TYPE[1], CELL_TYPE[0]}):
                    # Retraction into nucleus
                    grid[x, y] = CELL_TYPE[3]
                    grid[initial_in_pos] = CELL_TYPE[1]
                    Filaments[idx] = (Filaments[idx][0], (x, y))
                    swapped = True
                    #Check nucleus attachment
                    if not enhanced_attachment_check(grid, (x, y), CELL_TYPE[2], filament_idx=idx):
                        # Revert all changes
                        grid[x, y] = original_grid_xy
                        grid[initial_in_pos] = original_grid_initial
                        Filaments[idx] = (Filaments[idx][0], initial_in_pos)
                        return 0.0
                #If surrounded by nucleus
                elif neighbor_types.issubset({CELL_TYPE[2]}):
                    # Extension into cytoplasm
                    grid[x, y] = CELL_TYPE[3]
                    grid[initial_in_pos] = CELL_TYPE[2]
                    Filaments[idx] = (Filaments[idx][0], (x, y))
                    swapped = True
                    # Check nucleus attachment
                    if not enhanced_attachment_check(grid, (x, y), CELL_TYPE[2], filament_idx=idx):
                        # Revert all changes
                        grid[x, y] = original_grid_xy
                        grid[initial_in_pos] = original_grid_initial
                        Filaments[idx] = (Filaments[idx][0], initial_in_pos)
                        return 0.0
                else:
                    # Revert test move if conditions not met
                    grid[x, y] = original_grid_xy
                    break

    # ===========================
    # Connectedness test if no swap has occurred
    # ===========================
    #If no swap has occurred and the initial point is not a filament
    if not swapped and not simple_swap and old_type not in (CELL_TYPE[3], CELL_TYPE[4]):
        grid[x, y] = new_type
        # Connectivity checks
        if old_type == CELL_TYPE[1] and not is_connected_component(grid, CELL_TYPE[1], 30, ignore_coord=(x, y)):
            grid[x, y] = old_type
            return 0.0
        elif old_type == CELL_TYPE[2] and not is_connected_component(grid, CELL_TYPE[2], 10, ignore_coord=(x, y)):
            grid[x, y] = old_type
            return 0.0

    # ===========================
    # Energy calculation
    # ===========================
    new_energy = total_energy_cell(grid) + persist
    delta_energy = new_energy - old_energy

    # Metropolis criterion
    if delta_energy <= 0 or random.random() < np.exp(-delta_energy / TEMPERATURE):
        return delta_energy
    else:
        # Revert simple swap
        if simple_swap:
            grid[x, y], grid[nx, ny] = grid[nx, ny], grid[x, y]
            if filament_idx is not None:
                if (x, y) in [neighbor for neighbor_set in neighbors_outer for neighbor in neighbor_set]:
                    # This was an outer filament swap
                    Filaments[filament_idx] = (initial_out_pos, Filaments[filament_idx][1])
                else:
                    # This was an inner filament swap
                    Filaments[filament_idx] = (Filaments[filament_idx][0], initial_in_pos)
        #If not a simple swap, revert
        elif filament_idx is not None:
            grid[x, y] = old_type
            if 'initial_out_pos' in locals() and initial_out_pos is not None:
                grid[initial_out_pos] = CELL_TYPE[4]
                Filaments[filament_idx] = (initial_out_pos, Filaments[filament_idx][1])
            elif 'initial_in_pos' in locals() and initial_in_pos is not None:
                grid[initial_in_pos] = CELL_TYPE[3]
                Filaments[filament_idx] = (Filaments[filament_idx][0], initial_in_pos)
        else:
            #normal revert
            grid[x, y] = old_type
        return 0.0
    



#======================
#Visualization and main simulation function
#======================

#This array is used to track the COM of cell on the video
com_history = []
com_nucl_list = []

def visualize(grid, step, COM,com_history,COM_nuc,com_nucl_list):

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
   

     # Draw curved lines between filament using bezier curve
    for outer, inner in Filaments:
        x0, y0 = outer
        x1, y1 = inner

        # Create a control point between the two with a slight curve (midpoint raised)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.5  

        try:
            # Build spline path from 3 control points
            tck, _ = splprep([[x0, mx, x1], [y0, my, y1]], k=2)
            u_fine = np.linspace(0, 1, 100)
            smoothed = splev(u_fine, tck)
            plt.plot(smoothed[0], smoothed[1], color='black', linewidth=0.5, alpha=0.5)
        except ValueError:
            # Fallback: draw straight line if splprep fails
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


    #initialise the velocity and COM as queues with 0,0 for velocity and 0,[x0,y0]
    velocity = Queue(maxsize=2)
    COM_dyn = Queue(maxsize=2)

    velocity.put(np.array([[0,0]]))
    velocity.put(np.array([[1000,0]]))

    COM_dyn.put(np.array([0,0]))
    COM_dyn.put(COM(grid, 1,2))


    
    for step in range(MC_STEPS):

        #Persistance term based on flip attempt so within each step loop, we calculate the new persistnace term
        #Adapts the persistnace lagrange multiplier term from M. Scianna, L. Preziosi (2013) into a dynamic function
        #Queue actions are taken at the end of each flip, even if no change occurs, effectively setting v = 0 that step.
        l = length(grid)
        mu_pers = persistance_mu*(l/(2*r2) - 1)
        delta_v = velocity.queue[1]-velocity.queue[0]
        persist = mu_pers*np.linalg.norm(delta_v)**2



        # Perform multiple attempts per MCS
        for _ in range(ATTEMPTS_PER_STEP):   
            energy = attempt_flip(grid,persist)


        COM_dyn.get()
        COM_dyn.put(COM(grid,1,2))
        velocity.get()
        velocity.put(COM_dyn.queue[1]-COM_dyn.queue[0])


        current_energy = total_energy_cell(grid) + persist
        total_energy_list.append(current_energy)
        

        if step % interval == 0:
            visualize(grid, step,COM(grid,1,2) ,com_history,COM(grid,2),com_nucl_list)
            print(f'Step {step}: Energy = {current_energy}')

    com_cell,com_nuc = COM(grid, 1,2), COM(grid,2)

    visualize(grid, MC_STEPS,com_cell,com_history,com_nuc,com_nucl_list)
    print(f"The displacement of the cell COM in the x direction is  {com_cell[0]-x0}")
    print(f"The displacement of the nucleus COM in the x direction is {com_nuc[0]-x0}")

    

if __name__ == "__main__":
    run_simulation()
    

