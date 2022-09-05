import mujoco
import mujoco_viewer
import numpy as np

model = mujoco.MjModel.from_xml_path('./a1/xml/a1.xml')
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

kp = 80.
kd = 10.
# simulate and render
while True: 
    with open("mocap.txt","r") as filestream:
        for line in filestream:
            currentline = line.split(",")
            frame = currentline[0]
            t = currentline[1]
            target_joints = np.array([float(_q) for _q in currentline[2:14]])
            if viewer.is_alive:
                v = np.array(data.qvel[6:])
                q = np.array(data.qpos[7:])
                tau = kp * (target_joints-q) + kd * (-v)
                data.ctrl = tau
                mujoco.mj_step(model, data)
                viewer.render()
            else:
                break

# close
viewer.close()
