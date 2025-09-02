# Generates the mesh for flow past a cylinder, or more like half of
#   it, since the cylinder will only look like a bump
#   Actually accepts a meshsize parameter h instead of 
#   whatever meshsize/mesh density parameter Fenics uses lol


import gmsh
import sys
import meshio
from fenics import * 
import matplotlib.pyplot as plt


def main(h):
    gmsh.initialize() # must be done first

    # Create new gmsh model
    filename = "flow_cylinder_h_%.4e"%h
    filepath = "meshdata/" + filename # since this will usually be called by actual functions outside /meshdata
    
    gmsh.model.add(filename)

    # shortcut. Didn't know Python could do this lol
    factory = gmsh.model.geo

    height = 1.0
    width = 4.0
    cyl_rad = 0.5
    cyl_center = (1.5, 0.0)

    # b = bottom
    # l = left
    # t = top
    # r = right

    # mark corners of tube 
    bl_pt = factory.addPoint(0.0, 0.0, 0.0, h)
    tl_pt = factory.addPoint(0.0, height, 0.0, h)
    tr_pt = factory.addPoint(width, height, 0.0, h)
    br_pt = factory.addPoint(width, 0.0, 0.0, h)

    #mark corners of cylinder
    t_cyl_pt = factory.addPoint(cyl_center[0], cyl_center[1]+cyl_rad, 0.0, h)
    r_cyl_pt = factory.addPoint(cyl_center[0]+cyl_rad, cyl_center[1], 0.0, h)
    b_cyl_pt = factory.addPoint(cyl_center[0], cyl_center[1]-cyl_rad, 0.0, h)
    l_cyl_pt = factory.addPoint(cyl_center[0]-cyl_rad, cyl_center[1], 0.0, h)

    # need to encode center as gmsh point 
    cyl_center_pt = factory.addPoint(cyl_center[0], cyl_center[1], 0.0, h)
    
    # add boundary curves, going cc-wise
    l_wall = factory.addLine(bl_pt, tl_pt)
    t_wall = factory.addLine(tl_pt, tr_pt)
    r_wall = factory.addLine(tr_pt, br_pt)

    br_wall = factory.addLine(br_pt, r_cyl_pt)
    ne_cyl_arc = factory.addCircleArc(r_cyl_pt, cyl_center_pt, t_cyl_pt)
    nw_cyl_arc = factory.addCircleArc(t_cyl_pt, cyl_center_pt, l_cyl_pt)
    bl_wall = factory.addLine(l_cyl_pt, bl_pt)

    # add closed boundary loop
    boundary_loop = factory.addCurveLoop([l_wall, t_wall, r_wall, br_wall, ne_cyl_arc, nw_cyl_arc, bl_wall])
    
    # add domain interior (plane surface)
    domain_surface = factory.addPlaneSurface([boundary_loop])


    # give some convenient names to things. Need to synchronize first, I think? 
    gmsh.model.geo.synchronize()

    inflow_bndry_grp = factory.addPhysicalGroup(1, [l_wall])
    gmsh.model.setPhysicalName(1, inflow_bndry_grp, "Inflow")

    outflow_bndry_grp = factory.addPhysicalGroup(1, [r_wall])
    gmsh.model.setPhysicalName(1, outflow_bndry_grp, "Outflow")

    cyl_bndry_grp = factory.addPhysicalGroup(1, [ne_cyl_arc, nw_cyl_arc])
    gmsh.model.setPhysicalName(1, cyl_bndry_grp, "Cylinder")

    domain_grp = factory.addPhysicalGroup(2, [domain_surface])
    gmsh.model.setPhysicalName(2, domain_grp, "Domain")

    # Synchronize the CAD (.geo) entities with the model
    gmsh.model.geo.synchronize()

    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)

    # ... and save it to disk
    gmsh.write(filepath + ".msh")

    # Visualize mesh
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
        """"""

    # Always run this at the end
    gmsh.finalize()
    
    
    ##########################################################
    ####    Gmsh construction is over, now on to Fenics   ####
    ##########################################################

    # Now, we need to make sure this mesh plays nicely with FEniCS
    # This function was recommended by Dokken in:
    # https://jsdokken.com/src/pygmsh_tutorial.html, "Mesh generation and conversion with GMSH and PYGMSH"
    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        
        # Prune to 2D mesh if requested, ie ignore z component. mesh.prune_z_0() doesn't want to work
        points = mesh.points[:, :2] if prune_z else mesh.points
        
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]})
        
        return out_mesh
    
    # Read the recently created Gmsh mesh
    msh = meshio.read(filepath + ".msh")

    # Create 2D mesh. "True" flag since it's a 2D mesh
    triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
    meshio.write(filepath + "_triangle.xdmf", triangle_mesh)

    # Create 1D mesh
    line_mesh = create_mesh(msh, "line", prune_z=True)
    meshio.write(filepath + "_line.xdmf", line_mesh)

    #print(".xdmf files written successfully")
    
    # If you had had a 3D mesh, you would need a 3D mesh and a 2D mesh 
    #   - No 1D mesh needed for a 3D mesh, I don't think
    # Replace 'triangle' with 'tetra' and 'line' with 'triangle'. Do not prune

    # Bring it back into FEniCS
    mymesh = Mesh()

    # 2D triangles 
    with XDMFFile(filepath + "_triangle.xdmf") as infile:
        infile.read(mymesh)
    mvc_2d = MeshValueCollection("size_t", mymesh, 2) 

    with XDMFFile(filepath + "_triangle.xdmf") as infile:
        infile.read(mvc_2d, "name_to_read")
    mf_2d = cpp.mesh.MeshFunctionSizet(mymesh, mvc_2d)

    # 1D lines
    mvc_1d = MeshValueCollection("size_t", mymesh, 1)

    with XDMFFile(filepath + "_line.xdmf") as infile:
        infile.read(mvc_1d, "name_to_read")
    mf_1d = cpp.mesh.MeshFunctionSizet(mymesh, mvc_1d)

    # Save mesh as .h5 file for easy FEniCS access, filepath.h5/mesh
    outfile = HDF5File(MPI.comm_world, filepath + ".h5", 'w')
    outfile.write(mymesh, '/mesh')
    outfile.close()


    ##########################################################
    ####   FEniCS mesh cooperation is over, we are done   ####
    ##########################################################

    # Optional testing for FEniCS cooperation:

    """
    # Run this to recover the mesh from .h5 file for use in FEniCS:
    mesh2 = Mesh() #empty mesh
    infile = HDF5File(MPI.comm_world, filepath + ".h5", 'r')
    infile.read(mesh2, '/mesh', True) #for some reason, need this flag to import a mesh?
    infile.close()
    print("mesh recovered from .h5 file, numel = %d"%mesh2.num_cells())
    plot(mesh2, title = "mesh recovered from .h5 file, numel = %d"%mesh2.num_cells())
    plt.show()
    
    # make sure boundary is detected properly
    P_elem = FiniteElement("CG", triangle, 1)
    P = FunctionSpace(mesh2, P_elem)
    
    # Boundaries of domain. The issue with triangles is that they always undershoot circles, at least here

    class Cylinder(SubDomain):
        def inside(self, x, on_boundary):
            dist = (x[0]-cyl_center[0])*(x[0]-cyl_center[0]) + (x[1]-cyl_center[1])*(x[1]-cyl_center[1])
            return (on_boundary and dist <= cyl_rad*cyl_rad+h)
        
    class Inflow(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and near(x[0], 0.0))
    
    class Outflow(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and near(x[0], width))

    class Walls(SubDomain):
        def inside(self, x, on_boundary):
            ceiling = near(x[1], height)
            floor = near(x[1], 0.0)
            cyl_bndry = (x[0]-cyl_center[0])*(x[0]-cyl_center[0]) + (x[1]-cyl_center[1])*(x[1]-cyl_center[1]) < cyl_rad*cyl_rad+h
            return on_boundary and (ceiling or floor or cyl_bndry) and (not (near(x[0], 0.0) or near(x[0], width)))
    
    
    inflow_bc = DirichletBC(P, Constant(1.0),  Inflow())
    outflow_bc = DirichletBC(P, Constant(2.0), Outflow())
    cylinder_bc = DirichletBC(P, Constant(-1.0), Walls())
    
    test_func = Function(P)
    cylinder_bc.apply(test_func.vector())
    inflow_bc.apply(test_func.vector())
    outflow_bc.apply(test_func.vector())
    
    fig = plot(test_func, title = "testing bc")
    plt.colorbar(fig)
    plt.show()
    """

    
# In case it's called from command line
if __name__ == '__main__':
    main(float(sys.argv[1]))
