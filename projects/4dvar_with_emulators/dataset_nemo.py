import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
#import utils.utilities as ut
# import xarray as xr
# from torch.utils.data import DataLoader
# import models.lightning_models as lm


# -----------------------------------------------------------------------------
# Implementation date: 02/02/2024
# Modification date: 05/03/2024
# Purpose: create a new dataset class for NEMO in 2D
# Author: ntvminh286@gmail.com
# ----------------------------------------------------------------------------
class MyDatasetNEMO2D(Dataset):
    def __init__(self, ssh, uos, vos, params=None, normalize=False, datatype=torch.float32):
        self.datatype = datatype
        # declaration of public properties
        # information of domain discretization
        self.Ns = ssh.shape[0]  # number of cases (initial boundary conditions)
        self.nt = ssh.shape[1]  # number of discretization of time domain
        self.ny = ssh.shape[2]  # number of discretization in y direction of the space time
        self.nx = ssh.shape[3]  # number of discretization in x direction of the space time

        # assign data
        self.ssh = ssh[:, :, :, :]
        self.uos = uos[:, :, :, :]
        self.vos = vos[:, :, :, :]

        if params is not None:
            # self.zdom = params['zdom'][:, :, :, :]
            # self.cD = params['cD'][:, :, :, :]
            self.zdom = params['zdom']
            self.cD = params['cD']

        # normalize data
        # self._normalize = normalize
        self.normalize = {'turnOn': normalize,
                          'ssh': None,
                          'uos': None,
                          'vos': None,
                          'zdom': None,
                          'cD': None}

        if self.normalize["turnOn"]:
            self.ssh = self.normalize_data(self.ssh, 'ssh', normalization="unit")
            self.uos = self.normalize_data(self.uos, 'uos', normalization="unit")
            self.vos = self.normalize_data(self.vos, 'vos', normalization="unit")
            if params is not None:
                self.zdom = self.normalize_data(self.zdom, field_name='zdom', normalization="unit")
                self.cD = self.normalize_data(self.cD, field_name='cD', normalization="unit")
        
        if params is not None:
            self.X_input, self.X_target, self.params = self.__make_dataset(self.ssh,
                                                                    self.uos,
                                                                    self.vos,
                                                                    params=dict(zdom=self.zdom, cD=self.cD),
                                                                    datatype=self.datatype)
        else:
            self.X_input, self.X_target, self.params = self.__make_dataset(self.ssh,
                                                                    self.uos,
                                                                    self.vos,
                                                                    params=None,
                                                                    datatype=self.datatype)
        
        self.n_data = len(self.X_input)

    
    def normalize_data(self, x, field_name, normalization="unit"):
        if normalization == "std":
            x_normalize = (x - x.mean()) / x.std()
            self.normalize[field_name] = {'mean': x.mean(), 'std': x.std()}
        elif normalization == "unit" and field_name != 'zdom' and field_name != 'cD':
            # self.normalize["field"] = dict(name=field_name, type=normalization, mean=x.min(), std=x.max())
            x_normalize = (x - x.min()) / (x.max() - x.min())
            self.normalize[field_name] = {'max': x.max(), 'min': x.min()}
        elif normalization == "unit" and field_name == 'cD':
            x_normalize = (x - 0.0) / (0.01 - 0.0)
            self.normalize[field_name] = {'max': 0.01, 'min': 0.0}
        elif normalization == "unit" and field_name == 'zdom':
            x_normalize = (x - 0.0) / (100.0 - 0.0)
            self.normalize[field_name] = {'max': 100.0, 'min': 0.0}
        return x_normalize

    def denormalize_data(self, x, field_name, normalization="unit"):
        if normalization == "unit":
            x_max = self.normalize[field_name]['max']
            x_min = self.normalize[field_name]['min']
            denormal_x = x * (x_max - x_min) + x_min
        elif normalization == "std":
            x_mean = self.normalize[field_name]['mean']
            x_std  = self.normalize[field_name]['std']
            denormal_x = x * x_std + x_mean
        return denormal_x


    # private methods
    def __make_dataset(self, ssh, uos, vos, params=None, datatype=torch.float32):
        ssh_input, ssh_target = np.expand_dims(ssh[:, :-1, :, :].reshape(-1, self.ny, self.nx), axis=1), \
                                np.expand_dims(ssh[:,  1:, :, :].reshape(-1, self.ny, self.nx), axis=1)  # add a channel dimension
        uos_input, uos_target = np.expand_dims(uos[:, :-1, :, :].reshape(-1, self.ny, self.nx), axis=1), \
                                np.expand_dims(uos[:,  1:, :, :].reshape(-1, self.ny, self.nx), axis=1)  # add a channel dimension
        vos_input, vos_target = np.expand_dims(vos[:, :-1, :, :].reshape(-1, self.ny, self.nx), axis=1), \
                                np.expand_dims(vos[:,  1:, :, :].reshape(-1, self.ny, self.nx), axis=1)  # add a channel dimension

        X_input = np.concatenate((ssh_input, uos_input, vos_input), axis=1)
        X_target = np.concatenate((ssh_target, uos_target, vos_target), axis=1)

        if params is not None:
            zdom = params['zdom']
            cD = params['cD']
            zdom_input = np.expand_dims(zdom[:, :-1].reshape(-1), axis=1)
            cD_input = np.expand_dims(cD[:, :-1].reshape(-1), axis=1)
            params = np.concatenate((zdom_input, cD_input), axis=1)
        else:
            params = None
        
        return X_input, X_target, params

    # -----------------------------------------------------------------------------
    # Implementation date: 18/12/2023
    # Modified date: 02/01/2024
    # magic method
    # Purpose: get a specific data
    # ----------------------------------------------------------------------------
    def __getitem__(self, idx):
        X_input_at_idx = self.X_input[idx]
        X_target_at_idx = self.X_target[idx]
        if self.params is not None:
            params_at_idx = self.params[idx]
            return (
                        torch.tensor(X_input_at_idx, dtype=self.datatype),
                        torch.tensor(X_target_at_idx, dtype=self.datatype),
                        torch.tensor(params_at_idx, dtype=self.datatype)
            )
        else:
            return (
                        torch.tensor(X_input_at_idx, dtype=self.datatype),
                        torch.tensor(X_target_at_idx, dtype=self.datatype)                
            )    


        # tensors = [torch.tensor(x, dtype=self.datatype) for x in (X_input_at_idx, X_target_at_idx)]
        # return tensors

    # -----------------------------------------------------------------------------
    # Implementation date: 18/12/2023
    # magic method
    # Purpose: return the length of the dataset
    # ----------------------------------------------------------------------------
    def __len__(self):
        return self.n_data


# -----------------------------------------------------------------------------
# Implementation date: 16/10/2024
# Purpose: create a new dataset class for NEMO in 2D
# Author: ntvminh286@gmail.com
# ----------------------------------------------------------------------------
class DatasetOcean2D(Dataset):
    def __init__(self, data: dict,
                 params=None,
                 Ns: int = 1,
                 Nt: int = 1,
                 Ny: int = 1,
                 Nx: int = 1,
                 problem='tsunami',
                 normalize=False,
                 normlization_type='unit', 
                 datatype=torch.float32):
        # check problem is tsunami or swe or something else
        assert problem in ['tsunami', 'swe'], "Problem must be either 'tsunami' or 'swe'"
        assert data is not None, "Data must be provided"
        self.ssh = data['ssh']
        self.u = data['u']
        self.v = data['v']
        self.problem = problem
        if problem == 'tsunami':
            bathymetry = None
            maskBC = None
        elif problem == 'swe':
            bathymetry = np.repeat(data['bathymetry'][:, np.newaxis, :, :], Nt, axis=1)
            maskBC = np.repeat(data['maskBC'][:, np.newaxis, :, :], Nt, axis=1)
        self.datatype = datatype
        # declaration of public properties
        # information of domain discretization
        self.Ns = Ns  # number of cases (initial boundary conditions)
        self.Nt = Nt  # number of discretization of time domain
        self.Ny = Ny  # number of discretization in y direction of the space time
        self.Nx = Nx  # number of discretization in x direction of the space time

        if params is not None:
            self.zdom = params['zdom']
            self.cD = params['cD']
        else:
            params = None

        # normalize data
        # self._normalize = normalize
        self.normalize = {'turnOn': normalize,
                          'ssh': None,
                          'u': None,
                          'v': None,
                          'H': None,
                          'zdom': None,
                          'cD': None}

        if self.normalize["turnOn"]:
            self.ssh = self.normalize_data(self.ssh, 'ssh', normalization=normlization_type)
            self.u = self.normalize_data(self.u, 'u', normalization=normlization_type)
            self.v = self.normalize_data(self.v, 'v', normalization=normlization_type)
            if bathymetry is not None:
                if problem == 'swe':
                    normlization_type = 'm1to1'
                bathymetry = self.normalize_data(bathymetry, 'H', normalization=normlization_type)
            if params is not None:
                self.zdom = self.normalize_data(self.zdom, field_name='zdom', normalization="unit")
                self.cD = self.normalize_data(self.cD, field_name='cD', normalization="unit")
                params = dict(zdom=self.zdom, cD=self.cD)

        self.data = np.concatenate([np.expand_dims(i, axis=2) for i in (self.ssh, self.u, self.v)], axis=2)

        self.X_input, self.X_target, self.params, self.H, self.maskBC =  self.__make_dataset(self.ssh,
                                                                                            self.u,
                                                                                            self.v,
                                                                                            bathymetry,
                                                                                            maskBC,
                                                                                            params=params,
                                                                                            datatype=self.datatype)
        
        self.n_data = len(self.X_input)

    
    def normalize_data(self, x, field_name, normalization="unit"):
        if normalization == "m1to1": # scale data to [-1, 1]
            x_normalize = (x - x.min()) / (x.max() - x.min())
            x_normalize = 2 * x_normalize - 1
            self.normalize[field_name] = {'max': x.max(), 'min': x.min()}
        if normalization == "std":
            x_normalize = (x - x.mean()) / x.std()
            self.normalize[field_name] = {'mean': x.mean(), 'std': x.std()}
        elif normalization == "scale":
            scale_factor = np.sqrt((x**2).mean())  # RMS value
            x_normalize = x / scale_factor
            self.normalize[field_name] = {'scale_factor': scale_factor}
        elif normalization == "unit" and field_name != 'zdom' and field_name != 'cD':
            # self.normalize["field"] = dict(name=field_name, type=normalization, mean=x.min(), std=x.max())
            x_normalize = (x - x.min()) / (x.max() - x.min())
            self.normalize[field_name] = {'max': x.max(), 'min': x.min()}
        elif normalization == "unit" and field_name == 'cD':
            x_normalize = (x - 0.0) / (0.01 - 0.0)
            self.normalize[field_name] = {'max': 0.01, 'min': 0.0}
        elif normalization == "unit" and field_name == 'zdom':
            x_normalize = (x - 0.0) / (100.0 - 0.0)
            self.normalize[field_name] = {'max': 100.0, 'min': 0.0}
        return x_normalize

    def denormalize_data(self, x, field_name, normalization="unit"):
        if normalization == "unit" or normalization == "m1to1":
            x_max = self.normalize[field_name]['max']
            x_min = self.normalize[field_name]['min']
            denormal_x = x * (x_max - x_min) + x_min
        elif normalization == "std":
            x_mean = self.normalize[field_name]['mean']
            x_std  = self.normalize[field_name]['std']
            denormal_x = x * x_std + x_mean
        elif normalization == "scale":
            scale_factor = self.normalize[field_name]['scale_factor']
            # get n in the function x.mean()
            # n = x.size
            # dfdx = 1.0 / scale_factor - x @ x.T / ((scale_factor**3) * n)
            # denormal_x = x * dfdx
            denormal_x = x * scale_factor
        return denormal_x


    def __reshape_data_4_input(self, x):
        # index in is: Ns, Nt, Ny, Nx
        # index out is: Ns*Nt, 1, Ny, Nx
        return np.expand_dims(x[:, :-1, :, :].reshape(-1, self.Ny, self.Nx), axis=1)  # we remove the last time step for the input data
    
    def __reshape_data_4_target(self, x):
        # index in is: Ns, Nt, Ny, Nx
        # index out is: Ns*Nt, 1, Ny, Nx   
        return np.expand_dims(x[:,  1:, :, :].reshape(-1, self.Ny, self.Nx), axis=1)  # we remove the first time step for the target data
        
    
    # private methods
    def __make_dataset(self, ssh, u, v, H, maskBC, params=None, datatype=torch.float32):
        # Current index is: Ns, Nt, Ny, Nx
        # Output index is: Nt, 3, Ny, Nx, Ns  # 3 is the number of channels
        ssh_input, ssh_target = self.__reshape_data_4_input(ssh), self.__reshape_data_4_target(ssh)
        u_input, u_target = self.__reshape_data_4_input(u), self.__reshape_data_4_target(u)
        v_input, v_target = self.__reshape_data_4_input(v), self.__reshape_data_4_target(v)
        H_input = self.__reshape_data_4_input(H) if H is not None else None
        maskBC_input = self.__reshape_data_4_input(maskBC) if maskBC is not None else None
        X_input = np.concatenate((ssh_input, u_input, v_input), axis=1)
        X_target = np.concatenate((ssh_target, u_target, v_target), axis=1)
        if params is not None:
            zdom = params['zdom']
            cD = params['cD']
            zdom_input = np.expand_dims(zdom.T[:-1, :], axis=1)
            cD_input = np.expand_dims(cD.T[:-1, :], axis=1)
            params = np.concatenate((zdom_input, cD_input), axis=1)

        return X_input, X_target, params, H_input, maskBC_input

    # -----------------------------------------------------------------------------
    # Implementation date: 18/12/2023
    # Modified date: 02/01/2024
    # magic method
    # Purpose: get a specific data
    # ----------------------------------------------------------------------------
    def __getitem__(self, idx):
        X_input_at_idx = self.X_input[idx]
        X_target_at_idx = self.X_target[idx]
        params_at_idx = self.params[idx] if self.params is not None else None
        H_at_idx = self.H[idx] if self.H is not None else None
        maskBC_at_idx = self.maskBC[idx] if self.maskBC is not None else None

        # Helper function to convert to tensor or return an empty tensor if None
        def to_tensor_or_empty(value, dtype):
            if value is None:
                return torch.tensor([], dtype=dtype)
            return torch.tensor(value, dtype=dtype)


        return (
                    torch.tensor(X_input_at_idx, dtype=self.datatype),
                    torch.tensor(X_target_at_idx, dtype=self.datatype),
                    to_tensor_or_empty(params_at_idx, dtype=self.datatype),
                    to_tensor_or_empty(H_at_idx, dtype=self.datatype),
                    to_tensor_or_empty(maskBC_at_idx, dtype=self.datatype)
        )


        # tensors = [torch.tensor(x, dtype=self.datatype) for x in (X_input_at_idx, X_target_at_idx)]
        # return tensors

    # -----------------------------------------------------------------------------
    # Implementation date: 18/12/2023
    # magic method
    # Purpose: return the length of the dataset
    # ----------------------------------------------------------------------------
    def __len__(self):
        return self.n_data


def preprocess_data(loaded_data, TESTCASE, CONDI_NET=False, NORMALIZE=False, NORMALIZE_TYPE='unit'):
    import re
    if re.search(r'tsunami', TESTCASE, re.IGNORECASE):
        ssh = loaded_data['ssh']
        Ns = ssh.shape[0]
        Nt = ssh.shape[1]
        Ny = ssh.shape[2]
        Nx = ssh.shape[3]
        data = {'ssh': loaded_data['ssh'],
                'u': loaded_data['uos'],
                'v': loaded_data['vos']
                }
        if CONDI_NET == False:
            params = None
            ocean_ds = DatasetOcean2D(data=data, 
                                          params=params, 
                                          Ns=Ns,
                                          Nt=Nt,
                                          Ny=Ny,
                                          Nx=Nx, 
                                          problem='tsunami', 
                                          normalize=NORMALIZE,
                                          normlization_type=NORMALIZE_TYPE)
        else:
            params = dict(zdom=loaded_data['zdom'], cD=loaded_data['cD'])
            ocean_ds = DatasetOcean2D(data=data, 
                                          params=params, 
                                          Ns=Ns,
                                          Nt=Nt,
                                          Ny=Ny,
                                          Nx=Nx,
                                          problem='tsunami', 
                                          normalize=NORMALIZE,
                                          normlization_type=NORMALIZE_TYPE)
            
    elif re.search(r'swe', TESTCASE, re.IGNORECASE):
        ssh = loaded_data['z_vals']
        Ns = ssh.shape[0]
        Nt = ssh.shape[1]
        Ny = ssh.shape[2]
        Nx = ssh.shape[3]
        maskBC = np.ones_like(loaded_data['depth_profiles'])
        maskBC[:, 0, :] = maskBC[:, -1, :] = maskBC[:, :, 0] = maskBC[:, :, -1] = 0.0
        data = {'ssh': loaded_data['z_vals'],
                'u': loaded_data['u_vals'],
                'v': loaded_data['v_vals'], 
                'bathymetry': loaded_data['depth_profiles'], 
                'maskBC': maskBC
                }
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(data['bathymetry'][0, :, :].T, origin='lower')
        plt.colorbar()
        plt.show()

        print("ssh.shape = ", data['ssh'].shape)
        plt.figure()
        plt.imshow(data['ssh'][0, -1, :, :].T, origin='lower')
        plt.colorbar()
        plt.show()
        # return
        params = None
        ocean_ds = DatasetOcean2D(data=data, 
                                      params=params, 
                                      Ns=Ns,
                                      Nt=Nt,
                                      Ny=Ny,
                                      Nx=Nx,
                                      problem='swe',
                                      normalize=NORMALIZE,
                                      normlization_type=NORMALIZE_TYPE)
    return ocean_ds, ssh, params
# def test():
#     Ns = 2
#     nt = 10
#     nx = 5
#     ny = 5
#     np.random.seed(123)
#     ssh = np.random.rand(Ns, nt, ny, nx)
#     uos = np.random.rand(Ns, nt, ny, nx)
#     vos = np.random.rand(Ns, nt, ny, nx)

#     depth = np.ones((Ns, nt, ny, nx))
#     cD = np.ones((Ns, nt, ny, nx))

#     for i in range(Ns):
#         depth[i] = depth[i] * (i + 2)
#         cD[i] = cD[i] * (i + 1.5)/2.

#     # full dataset
#     full_dataset = MyDatasetNEMO2D(ssh, uos, vos, zdom=depth, cD=cD)

#     train_idx, val_idx = ut.generate_indices(len(full_dataset), train_frac=1.0, shuffle=False)
#     train_sampler, val_sampler = ut.generate_samplers(train_idx, val_idx)

#     # train_sampler, val_sampler = ut.generate_samplers(len(full_dataset), train_frac=1.0)
#     dataloader_train = DataLoader(dataset=full_dataset, shuffle=False, batch_size=1, sampler=train_sampler)
#     dataloader_val = DataLoader(dataset=full_dataset, shuffle=False, batch_size=1, sampler=val_sampler)

#     model = lm.CNNLightPeriodic()
#     model.train()
#     for (X_input, X_target, params) in dataloader_train:
#         data_pred = model(X_input, params)
#         print("X_input = ", X_input.shape)
#         print("X_target = ", X_target.shape)
#         print("data_pred = ", data_pred.shape)
#         print("params = ", params.shape)


# def integrate_nemo_data(data_size=1, batch_size=1):
#     ds = xr.open_mfdataset([f'data/TSUNAMI/TSUNAMI-{i:003}.nc' for i in range(1, data_size + 1)], concat_dim='combined_index', combine='nested')
#     ssh = ds['ssh'].values  # sea-surface height - (time_counter, y_grid_T, x_grid_T)
#     uos = ds['baro_u'].values  # zonal velocity - (time_counter, y_grid_U, x_grid_U)
#     vos = ds['baro_v'].values  # meridional velocity - (time_counter, y_grid_V, x_grid_V)

#     print("uos.shape = ", uos.shape)
#     print("vos.shape = ", vos.shape)
#     print("ssh.shape = ", ssh.shape)

#     ut.plot_x_2d(ssh[0, 0, :, :])  # plotting initial ssh of the 1st testcase

#     # initialize dataset and dataloader
#     full_dataset = MyDatasetNEMO2D(ssh, uos, vos, normalize=False)
#     train_sampler, val_sampler = ut.generate_samplers(len(full_dataset), train_frac=1.0, shuffle=False)
#     dataloader_train = DataLoader(dataset=full_dataset, shuffle=False, batch_size=50, sampler=train_sampler)
#     dataloader_val = DataLoader(dataset=full_dataset, shuffle=False, batch_size=50, sampler=val_sampler)
#     for (X_input, X_target) in dataloader_train:
#         print("X_input = ", X_input.shape)
#         print("X_target = ", X_target.shape)
#     return


# if __name__ == '__main__':
#     # ------------------------------------------------------------------------
#     # Just test if my dataloader works for random array or not
#     test()
#     # ------------------------------------------------------------------------
#     # Test with TSUNAMI data
#     # data_train = integrate_nemo_data(data_size=2, batch_size=1)
#     # ------------------------------------------------------------------------
