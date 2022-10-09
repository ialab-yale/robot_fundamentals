from turtle import heading
import meshcat
from meshcat import Visualizer
import meshcat.geometry as mc_geom
import meshcat.transformations as mc_trans
import numpy as np

l1 = 1.0
l2 = 1.0 
m1 = 1.0 
m2 = 1.0

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
        self._mass2 = self._link2["mass2"]

        self._wall = self["wall"]
        self._wall.set_object(
            mc_geom.Plane(1.0, 1.0),
            mc_geom.MeshLambertMaterial(
                    color=0x00ff00,
                    opacity=1.0,
                    reflectivity=0.8,
            )
        )
        self._wall.set_transform(
                    mc_trans.compose_matrix(
                        translate=[0,0,0], 
                        angles=[0.,np.pi/2,0.]
            )
        )
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

        self._link1.set_object(mc_geom.LineSegments(
            mc_geom.PointsGeometry(position=np.array([
            [0, 0, 0], [0, 0, -1.0]]).astype(np.float32).T,
            color=np.array([
            [1, 0, 0], [1, 0.6, 0],
            [0, 1, 0], [0.6, 1, 0],
            [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            ),
            mc_geom.LineBasicMaterial(vertexColors=True))
        )
        self._link2.set_object(mc_geom.LineSegments(
            mc_geom.PointsGeometry(position=np.array([
            [0, 0, 0], [0, 0, -1.0]]).astype(np.float32).T,
            color=np.array([
            [1, 0, 0], [1, 0.6, 0],
            [0, 1, 0], [0.6, 1, 0],
            [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            ),
            mc_geom.LineBasicMaterial(vertexColors=True))
        )
    def render(self, q):
        self._link1.set_transform(
            mc_trans.compose_matrix(
                translate=[0,0,2], 
                angles=[0,q[0],0]
            )
        )
        self._mass1.set_transform(mc_trans.translation_matrix([0,0,-l1]))
        self._link2.set_transform(
            mc_trans.rotation_matrix(q[1], [0,1,0])
        )
        self._mass2.set_transform(mc_trans.translation_matrix([0,0,-l2]))
