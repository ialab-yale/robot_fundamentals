import meshcat
from meshcat import Visualizer
import meshcat.geometry as mc_geom
import meshcat.transformations as mc_trans
import numpy as np

class BallViewer(Visualizer):
    def __init__(self, radius=1.0) -> None:
        Visualizer.__init__(self)
        self._body = self["ball"]
        self._body.set_object(
            mc_geom.Sphere(radius),
            mc_geom.MeshLambertMaterial(
                    # color=0x00ff00,
                    # opacity=0.5,
                    reflectivity=0.8,
                    map=mc_geom.ImageTexture(image=mc_geom.PngImage.from_file('./BeachBallColor.jpg'))
                    )
        )
    def render(self, q):
        x,z,th = q
        tf = mc_trans.compose_matrix(
                        translate=[x,0,z], 
                        angles=[0.,-th,0.]
            )
        self._body.set_transform(tf)