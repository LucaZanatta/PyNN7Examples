"""
Spiking Elementary Motion Detector (sEMD) example
See https://www.cit-ec.de/en/nbs/spiking-insect-vision for more details
"""

# imports
import spynnaker7.pyNN as p
import pylab
import csv
import datetime

# parameters
datum = datetime.datetime.now()

step = 0.1
p.setup(timestep=step)
n_neurons = 1
run_time = 150
cm = 0.25
i_offset = 0.0
tau_m = 20.0
tau_refrac = 1.0
current_decay = tau_syn_E = tau_syn_I = 30
v_reset = -85.0
v_rest = -65.0
v_thresh = -50.0

weight = 1
delay1 = 0.1
delay2 = 2.0

cell_params_lif = {'cm': cm, 'i_offset': i_offset, 'tau_m': tau_m,
                   'tau_refrac': tau_refrac, 'tau_syn_E': current_decay,
                   'tau_syn_I': current_decay, 'v_reset': v_reset,
                   'v_rest': v_rest, 'v_thresh': v_thresh}


# neuron populations
sEMD = p.Population(1, p.extra_models.IF_curr_exp_sEMD,
                    cell_params_lif, label="sEMD")
input_first = p.Population(1, p.SpikeSourceArray, {'spike_times': [[0]]},
                           label="input_first")
input_second = p.Population(1, p.SpikeSourceArray, {'spike_times': [[0]]},
                            label="input_second")


# projections
p.Projection(input_first, sEMD,
             p.OneToOneConnector(weights=weight, delays=delay1),
             target="excitatory")
p.Projection(input_second, sEMD,
             p.OneToOneConnector(weights=weight, delays=delay2),
             target="inhibitory")


# records
sEMD.record()  # spiketimes record
sEMD.record_v()  # membrane_voltage record
sEMD.record_gsyn()  # current record (modfied)

# run
p.run(run_time)

# get data
spikes = sEMD.getSpikes()  # read spikes
v = sEMD.get_v()  # read membrane voltage
currents = sEMD.get_gsyn()

print datum

# plots
time_voltage = [i[1] for i in v]
membrane_voltage = [i[2] for i in v if i[0] == 0]

time_current = [i[1] for i in currents]
curr_exc = [i[2] for i in currents if i[0] == 0]
curr_inh = [-i[3] for i in currents if i[0] == 0]

spike_time = [i[1] for i in spikes]
spike_id = [i[0] for i in spikes]

f, ((ax1, ax2), (ax3, ax4)) = pylab.subplots(2, 2, sharex='col')
ax1.plot(time_voltage, membrane_voltage)
ax1.set_title('membrane voltage over time')

ax2.text(0.2, 0.8, 'tau_syn: ')
ax2.text(0.2, 0.6, 'weights: ')
ax2.text(0.2, 0.4, 'delay1: ')
ax2.text(0.2, 0.2, 'delay2: ')

ax2.text(run_time/2, 0.8, current_decay)
ax2.text(run_time/2, 0.6, weight)
ax2.text(run_time/2, 0.4, delay1)
ax2.text(run_time/2, 0.2, delay2)

ax4.plot(time_current, curr_exc, time_current, curr_inh)
ax4.set_title('current over time')

ax3.plot(spike_time, spike_id, ".")
ax3.set_title('spikes over time')

pylab.show()

# export csv files
output_values = [membrane_voltage, time_voltage, curr_exc, curr_inh,
                 time_current, spike_id, spike_time]
testvalue = [cm, i_offset, tau_m, tau_refrac, current_decay,
             v_reset, v_rest, v_thresh]

csvfile = "output.csv"
csvfile_parameter = "output_parameter.csv"

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in output_values:
        writer.writerow(val)

with open(csvfile_parameter, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in testvalue:
        writer.writerow([val])

# end
p.end()
