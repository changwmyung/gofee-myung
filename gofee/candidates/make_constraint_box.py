from ase.io import write, read
import numpy as np
from abc import ABC, abstractmethod
from ase import Atoms, Atom
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.geometry import get_distances
from ase.units import Ang



class MakeBox():
    """ 
    This class used to make constraint_box, using bond length and slab.
    
    1)  specified_atoms: int; atomic number. 
        If you want to specify the atoms that may want to be an exception or to be a criterion, set this parameter.
        ex) object=Make_Box(specified_atoms=79,center_point=None,stoichiometry=stoichimetry,...),
            in this case, the box will be set based on this atoms.
    
    2)  center_point: list; 2-dimensional coordinates.
        If you want to specify the center position of the box, set this parameter.
        If not set the center_point, the box will be set based on the center of the slab.
        ex) center_point=[x_center,y_center]
            object=Make_Box(specified_atoms=None,center_point=center_point,...),
            in this case, the box will be set based on center_point.
    
    3)  stoichiometry: list; atomic numbers.
    
    4)  slab: Atoms object; Type of POSCAR or CONTCAR file in VASP
    
    5)  bl_factor: float; multiplier factor of bond length to set the box size.
        If you want to adjust the size, set this parameter.
        If bl_factor=None, the value is 3.0 that is default value. 
    
    6)  shrinkage: float; shirinkage factor
        If you want the box will shrink along to the loop, set this parameter.
        ex) object=Make_Box(...,stoichiometry=stoichimetry, shrinkage=0.1,...),   
    
    """     
    def __init__(self, stoichiometry, slab, 
                 specified_atoms=None, center_point=None, 
                 bl_factor=None, shrinkage=None, *args, **kwargs):
        
        self.specified_atoms = specified_atoms
        self.stoichiometry = stoichiometry
        self.slab = slab
        self.center_point = center_point
        
        if shrinkage is None:
            self.shrinkage = 0
        else:
            self.shrinkage = shrinkage

        if bl_factor is None:
            self.bl_factor = 3.0
        else:
            self.bl_factor = bl_factor
            

    def check_bond_length(self):
        """This funciton is used to check and get the longest bond length """

        Ntop = len(self.stoichiometry)
        num = np.reshape(self.stoichiometry,(Ntop,-1))

        slab_atoms=np.array([atom.number for atom in self.slab])
        slab_atoms_species = []

        for i in range(len(slab_atoms)):
            if slab_atoms[i] in slab_atoms_species:
                pass
            else:
                slab_atoms_species=np.append(slab_atoms_species,slab_atoms[i])

        num_total = np.append(num, slab_atoms_species)
        num_t = num_total.astype(int)

        r_cov = np.array([covalent_radii[n] for n in num_t])
        d = np.reshape(r_cov,(1,-1))
        b = np.sort(d)[:, ::-1]

        longest_bl=b[:,0]+b[:,1]

        return longest_bl
    
    
    def specify_pos(self):
        """This function is used to get the specified position's coordinate file if you want that."""

        indices=[atom.index for atom in self.slab if atom.number == self.specified_atoms]
        print(f'indices of specified_atoms = {indices}')
        specify_posi=[]
        specify_poscar=np.reshape(specify_posi,(-1,3))
        c = self.slab.get_positions()

        for i in indices:
            specify_poscar = np.vstack([specify_poscar,c[i]])
        print(f'specify_poscar = {specify_poscar}')
        print(f'len(specify_poscar) = {len(specify_poscar)}')

        return specify_poscar

    
    def make_box(self):
        """ 
        If not defined the specified_atoms parameter, 
        the box will be set the size of the box based on the center of the slab.

        If defined the center point, 
        the box will be set the size of the box based on the center point.
        
        If defined the specified_atoms parameter,
        the box will be set the size of the box based on the specified atoms
        """
        # Parameters for box setting
        slab = self.slab
        cell = slab.get_cell()
        k = self.shrinkage
        v = np.copy(cell)
        longest_bl = self.check_bond_length()
        template = slab.get_positions()
        
        slab_x_max = np.max(template[:,0])
        slab_y_max = np.max(template[:,1])
        slab_z_max = np.max(template[:,2])
        for i in range(3):
            for j in range(3):
                if i != j:
                    v[i][j]=0

        if self.specified_atoms is None:
            # Parameters for box setting
            # Set size of box
            v[0][0] = self.bl_factor*longest_bl
            v[1][1] = self.bl_factor*longest_bl
            # Set height of box
            v[2][2] = (1/2)*self.bl_factor*longest_bl
            
            if self.center_point is not None:
                center = self.center_point

                p0_x = center[0]-((1/2)*self.bl_factor*longest_bl)
                p0_y = center[1]-((1/2)*self.bl_factor*longest_bl)
                p0 = np.array((p0_x,p0_y,slab_z_max)) + (k*v[0][0],k*v[1][1],0)
                
                # Shrink box in v[0], v[1] and v[2] directions
                v[0][0] *= (1-2*k)
                v[1][1] *= (1-2*k)
                v[2][2] *= (1-k)
                
                # Make box
                box = [p0, v]
                return box

            else: # if center point is not defined,
                x_center = slab_x_max*0.5
                y_center = slab_y_max*0.5

                p0_x = x_center-((1/2)*self.bl_factor*longest_bl)
                p0_y = y_center-((1/2)*self.bl_factor*longest_bl)

                p0 = np.array((p0_x,p0_y,slab_z_max)) + (k*v[0][0],k*v[1][1],0)
                
                # Shrink box in v[0], v[1] and v[2] directions
                v[0][0] *= (1-2*k)
                v[1][1] *= (1-2*k)
                v[2][2] *= (1-k)
                
                # Make box
                box = [p0, v]
                return box

        else:
            # Parameters for box setting
            specify_poscar = self.specify_pos()

            x_max = np.max(specify_poscar[:,0])
            y_max = np.max(specify_poscar[:,1])
            z_max = np.max(specify_poscar[:,2])

            x_min = np.min(specify_poscar[:,0])
            y_min = np.min(specify_poscar[:,1])
            z_min = np.min(specify_poscar[:,2])

            # Set size of box
            v[0][0] = x_max-x_min+((2/3)*self.bl_factor*longest_bl)
            v[1][1] = y_max-y_min+((2/3)*self.bl_factor*longest_bl)
            # Set height of box
            v[2][2] = z_max-z_min+longest_bl
            
            x = x_min/3
            y = y_min/3
            
            p0 = np.array((x,y,z_min)) + (k*v[0][0],k*v[1][1],0)

            # Shrink box in v[0], v[1] and v[2] directions
            v[0][0] *= (1-2*k)
            v[1][1] *= (1-2*k)
            v[2][2] *= (1-k)
            
            # Make box
            box = [p0, v]
            return box

        print('v: ',v)
        print('box: ',box)
        print(f'center_point = {self.center_point}')



    def spherical_parameters(self):
        """ 
        This function is for get the radius and spherical_center parameters
        that used to the get_random_spherical() function of revised candidate_generation.py

        This will help to obtain parameters for a spherical surface 
        based on the center of the nanoparticles.  
        """

        specify_pos = self.specify_pos() 
        bondlength = self.check_bond_length() 

        x_max = np.max(specify_pos[:,0])
        y_max = np.max(specify_pos[:,1])
        z_max = np.max(specify_pos[:,2])

        x_min = np.min(specify_pos[:,0])
        y_min = np.min(specify_pos[:,1])
        z_min = np.min(specify_pos[:,2])

        radius_x = (x_max-x_min)+2*bondlength
        radius_y = (y_max-y_min)+2*bondlength
        radius_z = (z_max-z_min)+2*bondlength

        center_x = (x_max+x_min)/2
        center_y = (y_max+y_min)/2
        center_z = z_min

        #diameter = np.min(np.array((radius_x, radius_y))) #, radius_z
        diameter = np.max(np.array((radius_x, radius_y, radius_z)))
        radius = diameter/2
        spherical_center = np.array((center_x, center_y, center_z))

        return radius, spherical_center
        

