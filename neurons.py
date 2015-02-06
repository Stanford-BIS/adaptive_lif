from __future__ import division

import logging

import numpy as np

from nengo import AdaptiveLIFRate, LIF
from nengo.params import Parameter, NumberParam
from nengo.utils.compat import range

class AdaptiveLIF(AdaptiveLIFRate, LIF):
    """Adaptive spiking version of the LIF neuron model."""

    probeable = ['spikes', 'adaptation', 'voltage', 'refractory_time']

    def __init__(self, **kwargs):
        super(AdaptiveLIF, self).__init__(**kwargs)
        self.clip = -np.inf

    def _J_tspk(self, tspk):
        """Computes the input J that produces a steady state spike interval"""
        t0 = 1. / (1. - np.exp(-tspk / self.tau_rc))
        if self.tau_rc != self.tau_n:
            t1 = (self.inc_n * np.exp(-self.tau_ref / self.tau_n) *
                  (np.exp(-tspk / self.tau_n) - np.exp(-tspk / self.tau_rc)))
            t2 = ((1. - np.exp(-(self.tau_ref + tspk) / self.tau_n)) *
                  (self.tau_rc-self.tau_n))
        elif self.tau_rc == self.tau_n:  # Python doesn't know LHopital's Rule
            t1 = (-self.inc_n * tspk *
                  np.exp(-(self.tau_ref + tspk) / self.tau_rc))
            t2 = (self.tau_rc**2 *
                  (1 - np.exp(-(self.tau_ref + tspk) / self.tau_rc)))
        J = t0 * (1. - t1 / t2)
        return J

    def _num_rates(self, J, min_f=.001, max_f=None, max_iter=100, tol=1e-3):
        """Numerically determine the spiking adaptive LIF neuron tuning curve

        Uses the bisection method (binary search in CS parlance) to find the
        steady state firing rate for a given input

        Parameters
        ----------
        J : array-like of floats
            input
        min_f : float (optional)
            minimum firing rate to consider nonzero
        max_f : float (optional)
            maximum firing rate to seed bisection method with. Be sure that the
            maximum firing rate will indeed be within this bound otherwise the
            binary search will break
        max_iter : int (optional)
            maximium number of iterations in binary search
        tol : float (optional)
            tolerance of binary search algorithm in J. The algorithm terminates
            when maximum difference between estimated J and input J is within
            tol
        """
        f_ret = np.zeros_like(J)
        f_high = max_f
        if max_f is None:
            f_high = 1. / self.tau_ref
        f_low = min_f
        tspk_high = 1. / f_low - self.tau_ref
        tspk_low = 1. / f_high - self.tau_ref

        # check for J that produces firing rates below the minimum firing rate
        J_min = self._J_tspk(tspk_high)
        idx = J > J_min  # selects the range of J that produces spikes
        if not idx.any():
            return f_ret
        tspk_high = np.zeros(J[idx].shape) + tspk_high
        tspk_low = np.zeros(J[idx].shape) + tspk_low

        for i in xrange(max_iter):
            assert (tspk_low <= tspk_low).all(), 'binary search failed'
            tspk = (tspk_high + tspk_low) / 2.
            Jhat = self._J_tspk(tspk)
            max_diff = np.max(np.abs(J[idx]-Jhat))
            if max_diff < tol:
                break
            high_idx = Jhat > J[idx]  # where our estimate of J is too high
            low_idx = Jhat <= J[idx]  # where our estimate of J is too low
            tspk_high[low_idx] = tspk[low_idx]
            tspk_low[high_idx] = tspk[high_idx]
        f_ret[idx] = 1. / (self.tau_ref + tspk)
        return f_ret

    def rates(self, x, gain, bias):
        J = gain * x + bias
        out = self._num_rates(J)
        return out

    def gain_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to satisfy max_rates, intercepts.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        Parameters
        ----------
        max_rates : list of floats
            Maximum firing rates of neurons.
        intercepts : list of floats
            X-intercepts of neurons.
        """
        inv_tau_ref = 1. / self.tau_ref if self.tau_ref > 0 else np.inf
        if (max_rates > inv_tau_ref).any():
            raise ValueError(
                "Max rates must be below the inverse refractory period (%0.3f)"
                % (inv_tau_ref))

        tspk = 1. / max_rates - self.tau_ref
        x = self._J_tspk(tspk)
        gain = (1 - x) / (intercepts - 1.0)
        bias = 1 - gain * intercepts
        return gain, bias

    def step_math(self, dt, J, spiked, voltage, ref, adaptation):
        """Compute rates for input current (incl. bias)"""
        n = adaptation
        k = np.expm1(-dt / self.tau_n)
        inc = -k
        dec = k+1

        n *= dec  # decay adaptation
        LIF.step_math(self, dt, J - n, spiked, voltage, ref)
        n += inc * (self.inc_n * spiked)  # increment adaptation
