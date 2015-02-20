import logging
import numpy as np
import pytest

import nengo
from nengo_alif_steady_state import AdaptiveLIF
from nengo.utils.numpy import rms

logger = logging.getLogger(__name__)


def test_alif_neuron(Simulator, plt):
    """Test that the adaptive LIF dynamic model matches the predicted rates

    Tests a single neuron across multiple input currents
    """
    tau_n = .1
    inc_n = .1
    max_rates = np.array([200.])
    intercepts = np.array([.2])

    alif_neuron = AdaptiveLIF(tau_rc=.05, tau_ref=.002,
                              tau_n=tau_n, inc_n=inc_n)
    u_vals = np.array([
        0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    Ts = [
        .1, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    sim_rates = np.zeros_like(u_vals)
    num_rates = np.zeros_like(u_vals)
    for idx, (u, T) in enumerate(zip(u_vals, Ts)):
        net = nengo.Network()
        with net:
            stim = nengo.Node(u)
            net.ens = nengo.Ensemble(
                1, 1, neuron_type=alif_neuron,
                max_rates=max_rates, intercepts=intercepts)
            nengo.Connection(stim, net.ens.neurons, transform=np.array([[1.]]),
                             synapse=None)
            net.ps = nengo.Probe(net.ens.neurons, 'spikes')
        sim = Simulator(net)
        est_rate = net.ens.neuron_type.rates(
            u, sim.data[net.ens].gain, sim.data[net.ens].bias)
        sim.run(T)

        num_rates[idx] = est_rate
        spks = sim.data[net.ps]
        spk_times = np.nonzero(spks)[0]*sim.dt
        if len(spk_times) > 1:
            isi = np.diff(spk_times)
            sim_rates[idx] = 1. / np.mean(isi[-10:])
        if est_rate > 0.:
            rel_diff = abs(est_rate - sim_rates[idx]) / est_rate
            assert rel_diff < .01
        else:
            assert sim_rates[idx] == 0.

    plt.plot(u_vals, sim_rates, 'bo', ms=6, label='simulated rate')
    plt.plot(u_vals, num_rates, 'ro', ms=6, label='numerically estimated rate')
    plt.legend(loc='upper left')
    plt.xlabel('input')
    plt.ylabel('rate')


def test_alif_neurons(Simulator, plt, rng):
    """Test that the adaptive LIF dynamic model matches the predicted rates

    Tests an Ensemble of neurons at a single input value
    """
    dt = 0.001
    n = 100
    x = .5
    encoders = np.ones((n, 1))
    max_rates = rng.uniform(low=10, high=200, size=n)
    intercepts = rng.uniform(low=-1, high=1, size=n)

    net = nengo.Network()
    with net:
        ins = nengo.Node(x)
        ens = nengo.Ensemble(
            n, dimensions=1, encoders=encoders,
            max_rates=max_rates, intercepts=intercepts,
            neuron_type=AdaptiveLIF(tau_n=.1, inc_n=.1))
        nengo.Connection(ins, ens.neurons, transform=np.ones((n, 1)),
                         synapse=None)
        spike_probe = nengo.Probe(ens.neurons)
        voltage_probe = nengo.Probe(ens.neurons, 'voltage')
        adaptation_probe = nengo.Probe(ens.neurons, 'adaptation')
        ref_probe = nengo.Probe(ens.neurons, 'refractory_time')

    sim = Simulator(net, dt=dt)

    t_final = 3.0
    t_ss = 1.0  # time to consider neurons at steady state
    sim.run(t_final)

    n_select = rng.randint(n)  # pick a random neuron
    t = sim.trange()
    idx = t < t_ss
    plt.figure(figsize=(10, 6))
    plt.subplot(411)
    plt.plot(t[idx], sim.data[spike_probe][idx, n_select])
    plt.ylabel('spikes')
    plt.subplot(412)
    plt.plot(t[idx], sim.data[voltage_probe][idx, n_select])
    plt.ylabel('voltage')
    plt.subplot(413)
    plt.plot(t[idx], sim.data[adaptation_probe][idx, n_select])
    plt.ylabel('adaptation')
    plt.subplot(414)
    plt.plot(t[idx], sim.data[ref_probe][idx, n_select])
    plt.ylim([-dt, ens.neuron_type.tau_ref + dt])
    plt.xlabel('time')
    plt.ylabel('ref time')

    # check rates against analytic rates
    math_rates = ens.neuron_type.rates(
        x, *ens.neuron_type.gain_bias(max_rates, intercepts))
    idx = t >= t_ss
    spikes = sim.data[spike_probe][idx, :]
    sim_rates = (spikes > 0).sum(0) / (t_final - t_ss)
    logger.debug("ME = %f", (sim_rates - math_rates).mean())
    logger.debug("RMSE = %f",
                 rms(sim_rates - math_rates) / (rms(math_rates) + 1e-20))
    assert np.sum(math_rates > 0) > 0.5 * n
    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.001)

if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
