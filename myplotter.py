import pyvista as pv

class Picker:
    def __init__(self,plotter):
        self.plotter = plotter
    def __call__(self, mesh, idx):
        #print(mesh)
        #print(idx)
        #print(mesh.points[idx])
        self.plotter.set_focus(mesh.points[idx])
        

def mkplotter():
    pv.set_plot_theme('dark')
    plt = pv.Plotter()
    plt.add_axes()
    plt.show_grid()
    plt.enable_point_picking(Picker(plt), use_mesh=True,show_message=False)

    #plt.add_mesh(pv.Sphere())
    #plt.show()
    return plt
