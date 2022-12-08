This is an example of using below:

1) revised candidate_generation.py                                                   
  1-1) make_mol_structure() : a set molecule will be randomly positioned on the slab
  1-2) get_random_spherical() : a molecule will be randomly positioned to the spherical surface area based on the center of the slab or specified point(center parameter)
    
2) MakeBox class that involve the make_box, spherical_parameters, check_bond_length, and the except_pos functions     
  2-1) make_box() : can set the box parameter 
  2-2) spherical_parameters() : can set a radius and spherical_center point of spherical surface
