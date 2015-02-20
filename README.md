# nengo_alif_steady_state
Contains the AdaptiveLIF neuron model to be used in nengo

Installation
============
To install nengo_alif_steady_state::
  git clone https://github.com/Stanford-BIS/nengo_alif_steady_state
  cd nengo_alif_steady_state
  python setup.py develop
  
Usage
=====
  
Here is a minimal example of using this neuron model within Nengo

```
import nengo
import nengo_alif_steady_state

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: 0 if t < 0.1 else 1)
    ens = nengo.Ensemble(n_neurons=100, dimensions=1, 
                         neuron_type=nengo_alif_steady_state.AdaptiveLIF(
                             tau_rc=0.02,    # membrane time constant
                             tau_ref=0.002,  # refractory period
                             inc_n=0.1,      # magnitude of adaptive feedback
                             tau_n=0.5,      # time constant of adaptation
                             ))
    nengo.Connection(stim, ens)
    probe = nengo.Probe(ens, synapse=0.05)
    
sim = nengo.Simulator(model)
sim.run(1)

import pylab
pylab.plot(sim.trange(), sim.data[probe])
pylab.show()
```
