import numpy as np
from numpy import exp
import time
from mujoco_py import MjSim, MjSimPool, load_model_from_path
from scipy.stats import multivariate_normal
from scipy.signal import savgol_filter


class MultModelMPPI(object):

    ### Now I am going to pass in multiple models to see what happenes (need to speed this up)
    def __init__(self, models, task, frame_skip=1, horizon=10, num_trajectories=5, noise=0.2, lam=10.0):

        ### setup the simulator env
        self.models = models
        sim_pool = [] ## need to make a seperate list

        # model_probs = np.ones(len(models))
        # model_probs /= np.sum(model_probs)
        # self.model_probs = np.ones(no_trajectories * len(models))

        for i, model in enumerate(self.models): ### need to loop through each model now
            # self.model_probs[i*no_trajectories:i*no_trajectories+no_trajectories] = model_probs[i]
            for _ in range(num_trajectories):
                sim_pool.append(MjSim(model))

        self.model_probs = np.ones(len(sim_pool))
        self.model_probs /= np.sum(self.model_probs)

        self.pool = MjSimPool(sim_pool, nsubsteps=frame_skip) 
        ### simulators that are to be used
        ### this object holds a list of model + data structures for mujoco 
        ### think of each of these as a f(x, u) dynamics model 

        self.frame_skip = frame_skip

        ## also need to add the task into the system
        self.task = task

        self.data = [sim.data for sim in self.pool.sims] ### reference to the read-only data wrapper

        ### hopefully the number of states and actions are not changeing across each model
        self.num_states = self.models[0].nq + self.models[0].nv
        self.num_actions = self.models[0].nu
        self.num_sensors = self.models[0].nsensor ### I dont think i really need this now

        ### setup the parameters
        self.horizon                        = horizon
        self.num_trajectories_per_model     = num_trajectories
        self.num_tot_trajectories           = num_trajectories * len(self.models)
        self.noise                          = noise
        self.inv_sigma                      = np.eye(self.num_actions)/self.noise
        self.lam                            = lam

        self.sk = np.zeros((self.num_tot_trajectories, self.horizon)) ### entropy of the trajectory
        self.delta = np.zeros((self.num_tot_trajectories, self.horizon, self.num_actions))
        ### mean actions to be shifted as each action is computed and returned
        self.mean_actions = np.random.normal(0., self.noise, size=(self.horizon, self.num_actions))

    def __call__(self, state, predict_measurements=False): ### call to compute the control and return it

        ### pre-shift the mean actions -- this does nothing in the beginning
        self.mean_actions[:-1] = self.mean_actions[1:]
        self.mean_actions[-1] = np.random.normal(0., self.noise, size=(self.num_actions,))

        for sim in self.pool.sims: ### reset each of the simulators initial state
            sim.set_state(state)

        self.delta = np.random.normal(0., self.noise, size=self.delta.shape) ### create the random action perturbation

        for t in range(self.horizon): ### for each time step

            for k, data in enumerate(self.data): ### do each parameter fixing
                _ctrl = self.mean_actions[t] + self.delta[k, t, :]
                data.ctrl[:] = _ctrl
                self.sk[k,t] = self.task(data, _ctrl) \
                                + self.lam * np.dot(self.mean_actions[t], self.inv_sigma).dot(self.delta[k,t,:])
            #for _ in range(self.frame_skip):
            self.pool.step() ### step the simulator

        ### reverse cumulative sum in time
        self.sk = np.cumsum(self.sk[:, ::-1], axis=1)[:,::-1] ### in theory this should do what I think

        for t in range(self.horizon):### loop through each action and do the following
            sk = self.sk[:,t] ### sutract that out so you shift the weight
            self.sk[:,t] -= np.min(sk) ### sutract that out so you shift the weight
            _w = self.model_probs * exp(-self.sk[:,t]/self.lam) ### this should be of size no_trajectories
            _w /= np.sum(_w) ### normalize the weights
            self.mean_actions[t] += np.dot(_w, self.delta[:,t,:])
        self.mean_actions = savgol_filter(self.mean_actions, len(self.mean_actions)-1, 10, axis=0)

        ### I don't need to copy anything because the ctrl action would have been applied
        return self.mean_actions[0]




    def update_distribution(self, true_measurements, state, ctrl):

        for sim in self.pool.sims:
            sim.set_state(state)
            sim.data.ctrl[:] = ctrl
        #for _ in range(self.frame_skip):
        self.pool.step()

        for i, (data, model_prob) in enumerate(zip(self.data, self.model_probs)):
            simulated_measurements = data.sensordata[:]
            diff = true_measurements - simulated_measurements
            likelihood = np.prod(multivariate_normal.pdf(diff, 0., 0.01))
            self.model_probs[i] = model_prob * likelihood
        self.model_probs += 1e-10
        self.model_probs /= np.sum(self.model_probs)
