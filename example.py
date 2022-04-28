import napari

v = napari.Viewer()
v.dims.ndisplay = 3
v.open_sample("napari", "cells3d")
v.window.add_plugin_dock_widget("napari-manual-transforms")

napari.run()
