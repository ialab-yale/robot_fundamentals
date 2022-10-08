import meshcat
from meshcat import Visualizer
import meshcat.geometry as mc_geom
import meshcat.transformations as mc_trans
import numpy as np

l1 = 1.0
l2 = 1.0 
m1 = 1.0 
m2 = 1.0
def p1(q):
    th1, th2 = q
    return np.array([
        l1 * np.sin(th1),
        -l1 * np.cos(th1)
    ])
def p2(q):
    th1, th2 = q
    return p1(q) + np.array([
        l2 * np.sin(th1+th2),
        -l2 * np.cos(th1+th2)
    ])

class DoublePendViewer(Visualizer):
    def __init__(self) -> None:
        Visualizer.__init__(self)
        """
            Code below creates the tree structure 
            for the double pendulum visualization
        """
        self._link1 = self["link1"]
        self._mass1 = self._link1["mass1"]
        self._link2 = self._mass1["link1"]
        self._mass2 = self._link1["mass2"]

        self._mass1.set_object(
            mc_geom.Sphere(radius=0.1),
            mc_geom.MeshLambertMaterial(
                    # color=0x00ff00,
                    # opacity=0.5,
                    reflectivity=0.8,
                    map=mc_geom.ImageTexture(image=mc_geom.PngImage.from_file('./BeachBallColor.jpg'))
                    )
        )
        self._mass2.set_object(
            mc_geom.Sphere(radius=0.1),
            mc_geom.MeshLambertMaterial(
                    # color=0x00ff00,
                    # opacity=0.5,
                    reflectivity=0.8,
                    map=mc_geom.ImageTexture(image=mc_geom.PngImage.from_file('./BeachBallColor.jpg'))
                    )
        )
    def render(self, q):
        x1,y1 = p1(q)
        x2,y2 = p2(q)
        tf1 = mc_trans.compose_matrix(
                        translate=[x1,0,y1], 
                        angles=[0.,0.,0.]
            )
        tf2 = mc_trans.compose_matrix(
                        translate=[x2,0,y2], 
                        angles=[0.,0.,0.]
            )
        self._mass1.set_transform(tf1)
        self._mass2.set_transform(tf2)