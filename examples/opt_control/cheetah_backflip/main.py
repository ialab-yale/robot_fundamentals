import numpy as np
import mujoco
import mujoco_viewer
import time 

np.random.seed(10)


class MPPI(object):
    def __init__(self, xml_path, running_cost, time_skip=1,
        time_horizon=100, n_samples=25, var=0.2, lam=0.1) -> None:

        # create an additional mujoco model for mppi (not the same as the simulation)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        # create a list of mujoco data structures (this acts like a state)
        self.data = []
        for _ in range(n_samples):
            self.data.append(mujoco.MjData(self.model))

        # some book keeping variables
        self._n_samples    = n_samples
        self.running_cost  = running_cost
        self.lam = lam 
        self.var = var # variance ---> can be multi-dimensional or a scalar
        self.inv_var = 1./var
        self.tf  = time_horizon
        self.time_skip = time_skip
        self._n_ctrl = self.model.nu

        # the value function
        self.sk = np.zeros((self._n_samples, self.tf)) 
        # the shape of the random control noise for exploration
        self.eps_shape = (self._n_samples, self.tf, self._n_ctrl)
        # the resulting control
        self.u_mean = np.zeros((self.tf, self._n_ctrl))
    
    def calc_ctrl(self, init_state, max_iter=500):
        """
        This function calculates the optimized control through several iterations
        """
        for k in range(max_iter):

            # at each iteration, reset the state of each sample
            for data in self.data:
                data.qpos = init_state['qpos'].copy()
                data.qvel = init_state['qvel'].copy()
                mujoco.mj_forward(self.model, data)

            ### create the random ctrl noise
            eps = np.random.normal(0., self.var, size=self.eps_shape) 

            # make sure that we are not carrying any additional numbers around
            self.sk = self.sk*0.
            for t in range(self.tf):
                for n, data in enumerate(self.data):
                    # calculate the control + variation
                    _ctrl = self.u_mean[t] + eps[n,t,:]
                    # apply the control to the mujoco data structure for the nth sample
                    data.ctrl[:] = _ctrl 
                    # calculate the cost of that behavior
                    self.sk[n,t] = self.running_cost(data, _ctrl) + self.lam * self.inv_var * _ctrl @ _ctrl

                    # forward simulate using a model
                    # time skipping lets the model run at a faster frequency while 
                    # the controller runs at a slower rate
                    for tt in range(self.time_skip):
                        mujoco.mj_step(self.model, data)

            # compute the avg cost in case you want to look at it
            avg_cost = np.mean(self.sk)

            # this is fancy python arithmatic for computing the vaalue function 
            # line 15 in the mppi algorithm of the lecture notes
            self.sk = np.cumsum(self.sk[:,::-1], axis=1)[:,::-1]

            # this part updates the controls
            for t in range(self.tf):
                
                sk = self.sk[:,t]
                # shift down the value function so that it is numerically easy to use
                self.sk[:,t] -= np.min(sk)
                # compute the weights + little bit of white noise so there are no under flows
                _w = np.exp(-self.sk[:,t]/self.lam) + 1e-4
                # make w normalized i.e., a distribution
                _w /= np.sum(_w)
                # update u via importance sampling 
                self.u_mean[t] += np.dot(_w, eps[:,t,:])
            

def running_cost(data, u):
    rot = data.qpos[2]
    x_pos = data.qpos[0]
    x_vel = data.qvel[0]
    return   (x_pos)**2 + (rot-(-2*np.pi))**2

def run_open_loop():

    time_skip = 2
    xml_path = 'half_cheetah.xml'         

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    init_state = {
        'qpos' : data.qpos.copy(),
        'qvel' : data.qvel.copy()
    }
    mppi = MPPI(xml_path, running_cost, time_skip=time_skip)

    print('----calculating control please wait ----')
    mppi.calc_ctrl(init_state)
    print('----done! running control plan in openloop ----')

    viewer = mujoco_viewer.MujocoViewer(model, data)

    # execute in open loop
    while viewer.is_alive:
        data.qpos = init_state['qpos'].copy()
        data.qvel = init_state['qvel'].copy()
        mujoco.mj_forward(model, data)
        for ut in mppi.u_mean:
            data.ctrl = ut 
            for _ in range(time_skip):
                mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.01)


# execute in MPC 
def run_mpc():
    time_skip = 2
    xml_path = 'half_cheetah.xml'         

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    init_state = {
        'qpos' : data.qpos.copy(),
        'qvel' : data.qvel.copy()
    }
    # note we change the parameters of mppi so it can run a bit faster
    mppi = MPPI(xml_path, running_cost, time_horizon=60, n_samples=20, var=0.4, time_skip=time_skip)


    viewer = mujoco_viewer.MujocoViewer(model, data)

    # execute in open loop
    while viewer.is_alive:
        data.qpos = init_state['qpos'].copy()
        data.qvel = init_state['qvel'].copy()
        mujoco.mj_forward(model, data)
        for t in range(100):
                init_state = {
                    'qpos' : data.qpos.copy(),
                    'qvel' : data.qvel.copy()
                }
                # calculate one iteration step 
                mppi.calc_ctrl(init_state, max_iter=1)
                data.ctrl = mppi.u_mean[0].copy()
                for _ in range(time_skip):
                    mujoco.mj_step(model, data)
                # shift solution to warm start next iteration
                mppi.u_mean[:-1] = mppi.u_mean[1:]

                # render
                viewer.render()
                time.sleep(0.01)


if __name__=="__main__":

    # comment/uncomment the method you would like to try 

    #run_open_loop()
    run_mpc()
