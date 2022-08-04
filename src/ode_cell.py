import time
import numpy as np

import torch
import torch.nn as nn

# git clone https://github.com/rtqichen/torchdiffeq.git
from torchdiffeq import odeint as odeint

import src


#####################################################################################################
class ODECell(nn.Module):
    '''
        ODECell takes previous value, and current and previous time points and return new value.
        It calculates derivate of value using neural network and then calculates the value using
        the differential equation solver.
    '''
    def __init__(self, input_dim, latent_dim, rec_layers, units, nonlinear, ode_dropout, device = torch.device("cpu")):
        super(ODECell, self).__init__()

        ode_func_net = ODEFuncNet(latent_dim, latent_dim, ode_dropout,
                        n_layers = rec_layers, n_units = units, nonlinear = nonlinear).to(device)
        rec_ode_func = ODEFunc(input_dim = input_dim, latent_dim = latent_dim,
                               ode_func_net = ode_func_net, device = device).to(device)
        self.diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "euler", latent_dim,
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device).to(device)

        self.device = device

    def forward(self, prev_y, prev_t, t_i, minimum_step):
        # print('time values:', prev_t, t_i)
        if (abs(prev_t - t_i)) < minimum_step:
            time_points = torch.stack((prev_t, t_i))
            inc = self.diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

            assert(not torch.isnan(inc).any())
            ode_sol = prev_y + inc
            ode_sol = torch.stack((prev_y, ode_sol), 2).to(self.device)

            assert(not torch.isnan(ode_sol).any())
        else:
            n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())

            time_points = src.utils.linspace_vector(prev_t, t_i, n_intermediate_tp).to(prev_y.device)
            ode_sol = self.diffeq_solver(prev_y, time_points)

            assert(not torch.isnan(ode_sol).any())

        # if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
        #     print("Error: first point of the ODE is not equal to initial value")
        #     print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
        #     exit()
        # #assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)

        yi_ode = ode_sol[:, :, -1, :]

        return yi_ode#, time_points, ode_sol

#####################################################################################################

class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        # utils.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)
#########

'''
    Here, we define neural networks used in the ODE Function.
'''
class ODEFuncNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, ode_dropout, n_layers = 1,
                n_units = 100, nonlinear = nn.Tanh, std = 0.1) -> None:

        super(ODEFuncNet, self).__init__()

        # if ...
        self.net = PlainNet(n_inputs, n_outputs, n_layers, n_units, nonlinear, ode_dropout)
        self.net.init_network_weights(std)

    def forward(self, x):
        return self.net(x)

class PlainNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers = 1,
                        n_units = 100, nonlinear = nn.Tanh, ode_dropout=0.10, std = 0.1):
        super(PlainNet, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_units = n_units
        self.nonlinear = nonlinear

        layers = [nn.Linear(n_inputs, n_units)]
        for i in range(n_layers):
            layers.append(nonlinear())
            layers.append(nn.Dropout(ode_dropout)) # added dropout to ODE neural network
            layers.append(nn.Linear(n_units, n_units))

        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_outputs))

        self.net = nn.Sequential(*layers)

        self.init_network_weights(std)

    def init_network_weights(self, std = 0.1):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        return self.net(x)


#####################################################################################################


class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents,
            odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards = False):
        """
        # Decode the trajectory through ODE Solver
        """
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        n_dims = first_point.size()[-1]
        # print(first_point.device, time_steps_to_predict.device, self.ode_func.device)
        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        pred_y = pred_y.permute(1,2,0,3)

        assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
        assert(pred_y.size()[0] == n_traj_samples)
        assert(pred_y.size()[1] == n_traj)

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict,
        n_traj_samples = 1):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        # shape: [n_traj_samples, n_traj, n_tp, n_dim]
        pred_y = pred_y.permute(1,2,0,3)
        return pred_y


#####################################################################################################
