import meshcat
from meshcat import Visualizer
import meshcat.geometry as mc_geom
import meshcat.transformations as mc_trans
import numpy as np

class CartPendulumViewer(Visualizer):
    def __init__(self) -> None:
        Visualizer.__init__(self)
        self._cart  = self["cart"]
        self._pivot = self._cart["pivot"]
        self._pole = self._pivot["pole"]

        self._cart.set_object(
            mc_geom.Box([0.3, 0.5, 0.2]),
            # mc_geom.MeshLambertMaterial
            #     (
            #         color=0x00ff00,
            #         # opacity=0.5,
            #         reflectivity=0.8,
            #     )
        )

        self._pole.set_object(
            mc_geom.Box([0.05, 0.05, 1.0]),
            # mc_geom.MeshLambertMaterial
            #     (
            #         color=0x00ff00,
            #         # opacity=0.5,
            #         reflectivity=0.8,
            #     )
        )

        self._pole.set_transform(mc_trans.translation_matrix([0,0,0.5]))
        self._pivot.set_transform(mc_trans.rotation_matrix(np.pi/2.0, [1,0,0]))    
    def render(self, q):
        th, x = q
        self._cart.set_transform(
            mc_trans.translation_matrix([0,x,0])
        )
        self._pivot.set_transform(
            mc_trans.rotation_matrix(th, [1,0,0])
        )