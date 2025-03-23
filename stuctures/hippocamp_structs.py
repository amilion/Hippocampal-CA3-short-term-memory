from conex import *
from pymonntorch import *


class CA3Memory:
    def __init__(self, net: Neocortex):
        self.net= net
    
    def build_layer(self):
        ca3_e = NeuronGroup(
        net=self.net,
        size=15,
        behavior=prioritize_behaviors([
            SimpleDendriteStructure(),
            SimpleDendriteComputation(I_tau=2),
            LIF(
                R=10,
                tau=1,
                threshold=-50,
                v_rest=-63,
                v_reset=-65
            ),
            SpikeTrace(tau_s=20),
            NeuronAxon(),
            Fire(),
        ]) | {
            600: EventRecorder(["spikes"])    
        },
        tag="excitatory"
        )

        ca3_i = NeuronGroup(
            net=self.net,
            size=15,
            behavior=prioritize_behaviors([
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(
                    R=10,
                    tau=1,
                    threshold=-50,
                    v_rest=-63,
                    v_reset=-65
                ),
                SpikeTrace(tau_s=20),
                NeuronAxon(),
                Fire(),
            ]) | {
                600: EventRecorder(["spikes"])
            },
            tag = "inhibitory"
        )
        
        ex_to_ex = SynapseGroup(
            net=self.net,
            src=ca3_e,
            dst=ca3_e,
            behavior=prioritize_behaviors([
                SynapseInit(),
                WeightInitializer(mode="normal(0.01, 0.01)"),
                SimpleDendriticInput(),
                SimpleSTDP(a_minus=0.001, a_plus=0.01, w_min=0, w_max=1),
                ]
            ),
            tag="Proximal"
        )

        ex_to_inh = SynapseGroup(
            net=self.net,
            src=ca3_e,
            dst=ca3_i,
            behavior=prioritize_behaviors([
                SynapseInit(),
                WeightInitializer(mode="normal(0.8, 0.1)"),
                SimpleDendriticInput(),
                ]
            ),
            tag="Proximal"
        )

        inh_to_ex = SynapseGroup(
            net=self.net,
            src=ca3_i,
            dst=ca3_e,
            behavior=prioritize_behaviors([
                SynapseInit(),
                WeightInitializer(mode="normal(-0.8, 0.1)"),
                SimpleDendriticInput(),
                ]
            ),
            tag="GABA, Proximal"
        )

        inh_to_inh = SynapseGroup(
            net=self.net,
            src=ca3_i,
            dst=ca3_i,
            behavior=prioritize_behaviors([
                SynapseInit(),
                WeightInitializer(mode="normal(-0.4, 0.1)"),
                SimpleDendriticInput(),
                ]
            ),
            tag="GABA, Proximal"
        )