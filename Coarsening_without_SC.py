import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numba import njit
from datetime import datetime
from json import dump, load


class CoarseningWithoutSC:
    """
    This class defines the non-dimensional initial conditions of the simulation for the simple coarsening model
    without SC.
    Droplet allocation to SCs and positions do not affect the dynamics by definition and are thus not defined.
    Class evolves the droplets and concentrations, and dumps the log into a json file.
    Definition of the parameters based on SI-B.1b
    """

    def __init__(self, param_dict: dict):
        """
        Definition of the initial conditions based on the parameter dictionary
        :param param_dict: Parameter dictionary
        """

        self.param_dict = param_dict
        self.nu = self.param_dict['nu']  # sensitivity exponent nu
        self.nu_dn = self.param_dict['nu_dn']  # sensitivity exponent nu_DS
        self.phi_d = self.param_dict['phi_d']  # Affinity ratio between droplets and nucleoplasm
        self.n_d_init = self.param_dict['n_d_init']  # Number of initial droplets
        self.m_tot = self.param_dict['m_tot']  # Total amount of HEI10 in the cell
        self.m_d_init_mean = self.param_dict['m_d_init_mean']  # Average size of the initial droplets
        self.m_d_init_sd = self.param_dict['m_d_init_sd']  # relative standard deviation of initial droplets

        self.t_log = self.param_dict['t_log']  # List of times when system state is logged.
        if 0 not in self.t_log:
            # Ensure that the first element is 0
            self.t_log = np.append(self.t_log, 0)
        self.t_log = np.sort(self.t_log)
        self.t = 0  # Initial simulation time
        self.dt_init = self.param_dict['dt_init']  # Initial time step
        self.tol = self.param_dict['tol']  # Tolerance for the adaptive time-stepping
        self.sample_size = self.param_dict['sample_size']  # Number of samples of this simulation

        # To calculate the initial state we fix the average droplet size and initial number of droplets
        # and define the SC and nucleoplasm to be in equilibrium
        self.n_d = self.n_d_init
        self.m_n_init = self.m_tot - self.n_d * self.m_d_init_mean  # Total HEI10 amount without droplets

        # Define state variables
        self.m_n = np.ones(self.sample_size) * self.m_n_init  # HEI10 in nucleoplasm
        self.m_d = np.zeros((self.sample_size, self.n_d)) * np.nan  # Droplet size
        # Index of grid point of the respective SC the droplet belongs to
        for m in range(self.sample_size):
            # Initial size distribution of droplets, based on a clipped, normal distribution
            self.m_d[m, :] = np.clip(np.random.normal(1, self.m_d_init_sd, self.n_d), 0.1, 10)
            # Uniform droplet distribution along each SC
            self.m_d[m] *= self.m_d_init_mean / np.nanmean(self.m_d[m])  # fix the average droplet size for each sample

            self.m_n_log = None
            self.m_d_log = None
            self.n_d_log = None
            self.results = None

    def evolve(self):
        """
        Function to call the external numba functions running the explicit solvers.
        """
        now = datetime.now()
        self.t_log, self.m_n_log, self.m_d_log, self.n_d_log = \
            evolve_numba_explicit(self.t, self.t_log, self.dt_init, self.tol,
                                  self.nu, self.nu_dn, self.phi_d, self.m_n, self.m_d)
        # Write the results to a dictionary
        self.results = self.param_dict.copy()
        self.results['t_log'] = self.t_log
        self.results['m_n_log'] = self.m_n_log
        self.results['m_d_log'] = self.m_d_log
        self.results['n_d_log'] = self.n_d_log
        print('runtime: ', datetime.now() - now)
        return self.results

    def dump(self, path, filename, index=0):
        """
        Dump all log data to json file
        :param path: path to write data
        :param filename: filename to write data
        :param index: index of write data
        :return: results dictionary
        """
        # Write parameters, all relevant initial conditions and log arrays to dictionary
        results = self.param_dict.copy()
        results['t_log'] = self.t_log.tolist()
        results['m_n_log'] = self.m_n_log.tolist()
        results['m_d_log'] = self.m_d_log.tolist()
        results['n_d_log'] = self.n_d_log.tolist()
        path_filename = path + filename + '_' + str(index) + '.json'
        out_file = open(path_filename, "w")
        dump(results, out_file)
        out_file.close()
        return path_filename


def load_results(path, filename, index_list=np.array([0])):
    """
    Function to load the results of data with same parameter for a list of indices
    :param path: path to load data
    :param filename: filename to load data
    :param index_list: indices of load data
    :return: merged results dictionary
    """
    results = None
    for ii, index in enumerate(index_list):
        filename_idx = path + filename + '_' + str(index) + '.json'
        print('Load ', filename_idx)
        try:
            f = open(filename_idx, 'r')
            if results is None:
                results = load(f)
                results['m_n_log'] = np.array(results['m_n_log'])
                results['m_d_log'] = np.array(results['m_d_log'])
                results['n_d_log'] = np.array(results['n_d_log'])
                results['t_log'] = np.array(results['t_log'])
            else:
                results_idx = load(f)
                results['m_n_log'] = np.concatenate((results['m_n_log'], np.array(results_idx['m_n_log'])), axis=1)
                results['m_d_log'] = np.concatenate((results['m_d_log'], np.array(results_idx['m_d_log'])), axis=1)
                results['n_d_log'] = np.concatenate((results['n_d_log'], np.array(results_idx['n_d_log'])), axis=1)
            f.close()
        except Exception:
            print(filename_idx + '.json could not be loaded.')
            pass
    return results


@njit
def evolve_numba_explicit(t, t_log, dt_init, tol, nu, nu_dn, phi_d, m_n, m_d):
    """
    Numba function to run the adaptive time stepping and evolve the simulation state, and logging the data
    :param t: Current time
    :param t_log: List of times to log the system state
    :param dt_init: initial time step
    :param tol: Tolerance for adaptive time stepping
    :param nu to phi_d: Model parameters
    :param m_n, m_d: State variables
    :return: Log arrays
    """
    t_log_idx = 0  # index for writing simulation log
    sample_size = np.shape(m_n)[0]  # get sample size
    n_d = np.shape(m_d)[1]
    # Define log variables
    m_n_log = np.zeros((len(t_log), sample_size)) * np.nan
    m_d_log = np.zeros((len(t_log), sample_size, n_d)) * np.nan
    n_d_log = np.zeros((len(t_log), sample_size)) * np.nan

    # Write log
    t_log_idx, m_n_log, m_d_log, n_d_log, coarsening_complete = write_log(t_log_idx, m_n_log, m_n, m_d_log, m_d,
                                                                          n_d_log)
    dt = dt_init  # initial time step
    while t < t_log[-1] and not coarsening_complete:  # evolve as long as t < maximal value of t_log and N_D>1
        # Ensure that the values of the t_log list are exactly reached during the simulation
        if t + dt > t_log[t_log_idx]:
            dt = t_log[t_log_idx] - t
        # Evolve time step with adaptive time-stepping
        m_n, m_d, t, dt = evolve_adaptive_dt(t, dt, tol, nu, nu_dn, phi_d, m_n, m_d)
        # Write log
        if t >= t_log[t_log_idx]:
            t_log_idx, m_n_log, m_d_log, n_d_log, coarsening_complete = \
                write_log(t_log_idx, m_n_log, m_n, m_d_log, m_d, n_d_log)
    # After reaching at most one droplet per cell, stop evolving write the remaining log by copying
    m_n_log, m_d_log, n_d_log = write_remaining_log(t_log_idx, m_n_log, m_d_log, n_d_log)
    return t_log, m_n_log, m_d_log, n_d_log


@njit
def write_log(t_log_idx, m_n_log, m_n, m_d_log, m_d, n_d_log):
    """
    Writing the log of all state variables
    :param t_log_idx: current time log index
    :param m_n_log: log of HEI10 in nucleoplasms
    :param m_n: # current HEI10 in nucleoplasms
    :param m_d_log: log of droplet sizes
    :param m_d: current droplet sizes
    :param n_d_log: log of number of droplet per sample and SC
    :return: Log of state variables
    :return coarsening complete: bool that is True if all cells have at most one droplet left.
    """
    sample_size = np.shape(m_n)[0]
    n_d = np.shape(m_d)[1]
    # Log all state variables
    m_n_log[t_log_idx, :] = m_n
    m_d_log[t_log_idx, :, :] = m_d

    # Calculate droplet count for each SC in each sample
    for i in range(sample_size):
        droplet_count = 0
        for k in range(n_d):
            if m_d[i, k] > 0:
                droplet_count += 1
        n_d_log[t_log_idx, i] = droplet_count

    # Check whether coarsening is complete - this triggers stop of simulation
    if np.max(n_d_log[t_log_idx]) == 1.0:
        coarsening_complete = True
    else:
        coarsening_complete = False
    t_log_idx += 1
    return t_log_idx, m_n_log, m_d_log, n_d_log, coarsening_complete


@njit
def write_remaining_log(t_log_idx, m_n_log, m_d_log, n_d_log):
    """
    Write remaining log. Log variables compare write_log()
    return: Log of state variables
    """
    if t_log_idx < np.shape(n_d_log)[0] - 1:
        m_n_log[t_log_idx:, :] = m_n_log[t_log_idx - 1, :]
        m_d_log[t_log_idx:, :, :] = m_d_log[t_log_idx - 1, :, :]
        n_d_log[t_log_idx:, :] = n_d_log[t_log_idx - 1, :]
    return m_n_log, m_d_log, n_d_log


@njit
def evolve_adaptive_dt(t, dt, tol, nu, nu_dn, phi_d, m_n, m_d):
    """
    Numba function to evolve the state variables, with adaptive time-stepping
    :return: updated state variables
    """
    flux_m_d = compute_flux_explicit(dt, nu, nu_dn, phi_d, m_n, m_d)
    flux_m_d_0 = compute_flux_explicit(dt / 2, nu, nu_dn, phi_d, m_n, m_d)
    flux_m_n_0 = -np.sum(flux_m_d_0, axis=1)
    flux_m_d_1 = compute_flux_explicit(dt, nu, nu_dn, phi_d, m_n + flux_m_n_0, m_d + flux_m_d_0)
    # Adaptive time-stepping based on approach shown in https://en.wikipedia.org/wiki/Adaptive_step_size
    error = np.max(np.abs((flux_m_d_1 + flux_m_d_0) - flux_m_d))
    if error < tol:
        # Evolve droplets
        m_n -= np.sum(flux_m_d_1 + flux_m_d_0, axis=1)
        m_d += flux_m_d_1 + flux_m_d_0
        t = t + dt
    if error > 0:
        dt_factor = np.sqrt(tol / (2 * error))
        if dt_factor < 0.3:
            dt_factor = 0.3
        elif dt_factor > 2:
            dt_factor = 2.0
    else:
        dt_factor = 2.0
    dt = 0.9 * dt * dt_factor
    return m_n, m_d, t, dt


@njit
def compute_flux_explicit(dt, nu, nu_dn, phi_d, m_n, m_d):
    """
    Compute and return fluxes
    """
    sample_size = np.shape(m_n)[0]
    n_d = np.shape(m_d)[1]
    flux_dn = np.zeros_like(m_d)

    for i in range(sample_size):
        for j in range(n_d):
            if m_d[i, j] > 0:
                phi_m_d_a = phi_d * m_d[i, j] ** (-nu)
                # Exchange between nucleoplasm and droplets
                if nu_dn == 0.0:
                    flux_dn[i, j] = dt * (m_n[i] - phi_m_d_a)
                else:
                    flux_dn[i, j] = dt * (m_n[i] - phi_m_d_a) * m_d[i, j] ** nu_dn
                # check for vanishing droplets
                if m_d[i, j] + flux_dn[i, j] < 0.0:  # if droplet vanishes
                    # set volume to zero to avoid negative values and re-distribute exchange
                    flux_dn[i, j] = -  m_d[i, j]
    return flux_dn


def plot_time_development(sim_results, save_to_file=False, path='', filename=''):
    fig, ax = plt.subplots(figsize=(4, 3.0))
    ax2 = ax.twinx()
    sample_size = sim_results['sample_size']
    t_log = sim_results['t_log']

    # Plot average droplet size, and sem of average size over samples
    m_d_log = sim_results['m_d_log']
    m_d_log[m_d_log == 0.0] = np.nan
    mean_volumes = np.nanmean(m_d_log, axis=(1, 2))
    sem_volumes = np.std(np.nanmean(m_d_log, axis=2), axis=1) / np.sqrt(sample_size)
    ax.plot(t_log, mean_volumes, '-', color='C0', linewidth=0.5, label=r'Average droplet size $\bar{V}/V_0$')
    ax.fill_between(t_log, mean_volumes - sem_volumes, mean_volumes + sem_volumes, color='C0', alpha=0.3)

    # Plot average amount of HEI10 in nucleoplasm
    m_n_log = sim_results['m_n_log']
    mean_m_n = np.mean(m_n_log, axis=1)
    sem_m_n = np.std(m_n_log, axis=1) / np.sqrt(sample_size)
    ax.plot(t_log, mean_m_n, '-', color='C1', linewidth=0.5, label=r'Nucleoplasm $c_\mathrm{N}/c_0$')
    ax.fill_between(t_log, mean_m_n - sem_m_n, mean_m_n + sem_m_n, color='C1', alpha=0.3)

    # Plot average number of remaining droplets in cell
    n_d_log = sim_results['n_d_log']
    mean_n_d = np.mean(n_d_log, axis=1)
    sem_n_d = np.std(n_d_log, axis=1) / np.sqrt(sample_size)
    ax2.plot(t_log, mean_n_d, '-', color='C3', linewidth=0.5, label=r'Average droplet count $\bar{N}$')
    ax2.fill_between(t_log, mean_n_d - sem_n_d, mean_n_d + sem_n_d, color='C3', alpha=0.3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_yscale('log')
    ax.set_xlabel(r'Time $t/t_0$', fontsize=8)
    ax.set_ylabel(r'Concentrations / droplet size', fontsize=8)
    ax2.set_ylabel(r'Average droplet count $\bar{N}$', fontsize=8)

    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    ax2.tick_params(axis='both', which='minor', labelsize=6)
    ax2.tick_params(axis='both', which='major', labelsize=6)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6)

    plt.subplots_adjust(right=0.80, top=0.97, left=0.15, bottom=0.15)
    if save_to_file:
        plot_filename = path + filename + '.pdf'
        plt.savefig(plot_filename, dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    t_log = np.sort(np.concatenate((np.array([0]), 10 ** np.arange(-1.0, 4.01, 0.1))))

    path = os.getcwd() + '/'

    filename = 'model_without_SC'

    # Define parameters of model for system without SC, based on non-dimensional model in S2.A.2
    param = dict(nu=1 / 3,  # Size-dependency of HEI10 affinity to droplets
                 nu_dn=1 / 3,  # Size-dependency of exchanges between droplets and nucleoplasm
                 phi_d=1e-2,  # Affinity ratio between droplets and nucleoplasm
                 m_tot=5,  # Total amount of HEI10 in the cell
                 n_d_init=400,  # Number of initial droplets
                 m_d_init_mean=1e-3,  # Average initial droplet size
                 m_d_init_sd=1e-2,  # Standard deviation of initial droplet size
                 t_log=t_log,  # List of times to log the system state
                 dt_init=1e-3,  # Initial time-step
                 tol=1e-6,  # Tolerance for adaptive-time-stepping
                 sample_size=5)  # Sample Size

    sim1 = CoarseningWithoutSC(param)
    # Evolve system
    sim1.evolve()
    # Dump data to file
    sim1.dump(path, filename)
    # Plot the time development of the averages
    plot_time_development(sim1.results, save_to_file=True, path=path, filename=filename)
    # Load results from file
    # results = load_results(path, filename)
