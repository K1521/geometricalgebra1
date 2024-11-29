import pyvista as pv

class Picker:
    def __init__(self,plotter):
        self.plotter = plotter
    #def __call__(self, mesh, idx):
    def __call__(self, point,picker):
        #print(mesh)
        #print(idx)
        #print(mesh.points[idx])
        self.plotter.set_focus(picker.GetDataSet().points[picker.GetPointId()])
        #print(idx)
        

def mkplotter(dark=True):
    if dark:
        pv.set_plot_theme('dark')
    plt = pv.Plotter()
    plt.add_axes()
    plt.show_grid()
    #plt.enable_point_picking(Picker(plt), use_mesh=True,show_message=False)
    plt.enable_point_picking(Picker(plt), picker='point',use_picker=True,show_message=False)

    #plt.add_mesh(pv.Sphere())
    #plt.show()
    return plt
