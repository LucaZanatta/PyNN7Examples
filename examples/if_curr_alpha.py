import spynnaker7.pyNN as p
import pylab
p.setup(0.1)

pop_src1 = p.Population(1, p.SpikeSourceArray,
                        {'spike_times': [[5, 15, 20, 30]]}, label="src1")

pop_ex = p.Population(1, p.IF_curr_alpha, {}, label="test")
pop_ex.set("tau_syn_E", 2)
pop_ex.set("tau_syn_I", 4)

# define the projection
exc_proj = p.Projection(pop_src1, pop_ex,
                        p.OneToOneConnector(weights=1, delays=1),
                        target="excitatory")
inh_proj = p.Projection(pop_src1, pop_ex,
                        p.OneToOneConnector(weights=1, delays=20),
                        target="inhibitory")

pop_ex.record_gsyn()
pop_ex.record_v()
p.run(50)

v = pop_ex.get_v()
curr = pop_ex.get_gsyn()

p.end()

pylab.subplot(2, 1, 1)
pylab.xlabel('Time/ms')
pylab.ylabel('v')
pylab.plot(v[:, 1], v[:, 2])
pylab.subplot(2, 1, 2)
pylab.xlabel('Time/ms')
pylab.ylabel('current')
pylab.plot(curr[:, 1], curr[:, 2])
pylab.plot(curr[:, 1], curr[:, 3])
pylab.show()



