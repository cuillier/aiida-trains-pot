"""
Script for running active learning with magnetic MACE models.

Assumes you have generated a large dataset of magnetic configurations ex-situ,
and want to use uncertainty quantification to 
"""


from aiida.orm import load_code, load_node, load_computer, load_group, Str, Dict, List, Int, Bool, Float
from aiida import load_profile
from aiida.engine import submit
from aiida.plugins import WorkflowFactory, DataFactory
from aiida_quantumespresso.common.types import SpinType
from ase.io import read
import yaml
import os
from aiida_trains_pot.utils.restart import models_from_trainingwc,  models_from_aiidatrainspotwc
from aiida_trains_pot.utils.generate_config import generate_lammps_md_config

load_profile('develop')

PESData     = DataFactory('pesdata')
KpointsData = DataFactory("core.array.kpoints")
TrainsPot   = WorkflowFactory('trains_pot.workflow')

####################################################################
#                     START MACHINE PARAMETERS                     #
####################################################################

QE_code                 = load_code('qe7.4-gpu-pw@midway3')
MACE_train_code         = load_code('symmetrix_train@midway3')
MACE_preprocess_code    = load_code('symmetrix_preprocess@midway3')
MACE_postprocess_code   = load_code('symmetrix_postprocess@midway3')
LAMMPS_code             = load_code('symmetrix_lammps_ampere@midway3')
EVALUATION_code         = load_code('symmetrix_committee_evaluation')

ACCOUNT = "pi-gagalli"
CPU_PARTITION = "gagalli-csl2"
GPU_PARTITION = "gagalli-gpu"

QE_machine = {
'time'                             : "01:00:00",
'nodes'                            : 1,
'gpu'                              : 1,
'taskpn'                           : 1,
'cpupt'                            : 8,
'account'                          : ACCOUNT,
'partition'                        : GPU_PARTITION,
}

MACE_machine = {
'time'                             : "02:00:00",
'nodes'                            : 1,
'gpu'                              : 1,
'taskpn'                           : 1,
'cpupt'                            : 8,
'account'                          : ACCOUNT,
'partition'                        : GPU_PARTITION,
}

LAMMPS_machine = {
'time'                             : "01:00:00",
'nodes'                            : 1,
'gpu'                              : 1,
'taskpn'                           : 1,
'cpupt'                            : 1,
'account'                          : ACCOUNT,
'partition'                        : GPU_PARTITION,
}

EVALUATION_machine = {
'time'                             : "04:00:00",
'nodes'                            : 1,
'gpu'                              : 1,
'taskpn'                           : 1,
'cpupt'                            : 8,
'account'                          : ACCOUNT,
'partition'                        : GPU_PARTITION,
}

####################################################################
#                      END MACHINE PARAMETERS                      #
####################################################################

def get_memory(mem):
    if mem.find('MB') != -1:
        mem = int(mem.replace('MB',''))*1024
    elif mem.find('GB') != -1:
        mem = int(mem.replace('GB',''))*1024*1024
    elif mem.find('KB') != -1:
        mem = int(mem.replace('KB',''))
    return mem

def get_time(time):
    time = time.split(':')
    time_sec=int(time[0])*3600+int(time[1])*60+int(time[2])
    return time_sec

QE_time = get_time(QE_machine['time'])
MACE_time = get_time(MACE_machine['time'])
LAMMPS_time = get_time(LAMMPS_machine['time'])
EVALUATION_time = get_time(EVALUATION_machine['time'])

script_dir = os.path.dirname(os.path.abspath(__file__))

###############################################
# Input structures
###############################################

input_atoms = read(os.path.join(script_dir, 'SiC.xyz'),
                   index=':', format='extxyz')
input_structures = PESData(input_atoms)

###############################################
# Setup TrainsPot worflow
###############################################

builder                             = TrainsPot.get_builder(abinitiolabeling_code     = QE_code,
                                                            abinitiolabeling_protocol = 'stringent',
                                                            pseudo_family             = 'PseudoDojo/0.5/PBE/SR/stringent/upf',
                                                            md_code                   = LAMMPS_code,
                                                            dataset                   = input_structures,
                                                            )

# Restart from previous aiida-trains-pot workchain
#models_from_aiidatrainspotwc(builder, 9422) 

# These flags control what is done on the first iteration 
# If all are false, only do committee evaluation
builder.do_dataset_augmentation     = Bool(True)
builder.do_ab_initio_labelling      = Bool(True)
builder.do_training                 = Bool(True)
builder.do_exploration              = Bool(True)
builder.max_loops                   = Int(2)

#builder.dataset = load_node(112466) + load_node(114182)

# For skipping to the exploration step, using a previous TrainingWorkChaini
#models_from_trainingwc(builder, 114098, get_labelled_dataset=False, get_config=False) # TrainingWorkChain node

# For skipping to the exploration step, using existing models
#builder.models_lammps = {"pot_1": load_node(2536)}  # mace-mh-0 omat_pbe head, La-Sr-Co-O system, symmetrix format
#builder.models_ase = {"pot_1": load_node(2537)}     # ASE format

#builder.models_lammps = {"pot_1":load_node(85984), "pot_2":load_node(85995), "pot_3":load_node(86006), "pot_4":load_node(86017)} ## MACE potentials compiled for LAMMPS
#builder.models_ase = {"pot_1":load_node(85985), "pot_2":load_node(85996), "pot_3":load_node(86007), "pot_4":load_node(86018)} ## MACE potentials compiled for ASE

# For skipping to the committee evaluation step, using the PESData from an ExplorationWorkChain
#builder.explored_dataset = load_node(13875) # PESData node

###############################################
# Thresholds on committe evaluation to select
# structures to be labelled
###############################################
builder.thr_energy          = Float(0.002) # eV/atom
builder.thr_forces          = Float(0.100) # eV/Angstrom
builder.thr_stress          = Float(0.010) # eV/A3
builder.max_selected_frames = Int(8)       # Number of DFT calculations per iteration

###############################################
# Setup dataset augmentation
###############################################

builder.dataset_augmentation.do_rattle_strain_defects           = Bool(True)
builder.dataset_augmentation.do_input                           = Bool(True)
builder.dataset_augmentation.do_isolated                        = Bool(False)
builder.dataset_augmentation.do_clusters                        = Bool(False)
builder.dataset_augmentation.do_slabs                           = Bool(False)
builder.dataset_augmentation.do_replication                     = Bool(True)
builder.dataset_augmentation.do_check_vacuum                    = Bool(False)
builder.dataset_augmentation.do_substitution                    = Bool(False)  # Swapping
builder.dataset_augmentation.do_magnetic                        = Bool(False)
builder.dataset_augmentation.do_alloys                          = Bool(False)

# Rattle, strain, defect augmentations
builder.dataset_augmentation.rsd.params.rattle_fraction         = Float(0.20)   # Displace by (at most) this fraction of the equilibrium bond distance
builder.dataset_augmentation.rsd.params.max_tensile_strain      = Float(0.02)   # For LCO, approx. +10/-5% -> +- 50 kbar
builder.dataset_augmentation.rsd.params.max_compressive_strain  = Float(0.02)
builder.dataset_augmentation.rsd.params.n_configs               = Int(8)        # Number of RSD augmented calculations to do (per input structure)
builder.dataset_augmentation.rsd.params.frac_vacancies          = Float(1/8)
builder.dataset_augmentation.rsd.params.vacancies_per_config    = Int(1)

builder.dataset_augmentation.replicate.min_dist                 = Float(4.0)
builder.dataset_augmentation.replicate.max_atoms                = Int(8 * 2**3)

# Others
#builder.dataset_augmentation.clusters.n_clusters                = Int(80)
#builder.dataset_augmentation.clusters.max_atoms                 = Int(30)
#builder.dataset_augmentation.clusters.interatomic_distance      = Float(1.5)
#builder.dataset_augmentation.slabs.miller_indices               = List([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1]])
#builder.dataset_augmentation.slabs.min_thickness                = Float(10)
#builder.dataset_augmentation.slabs.max_atoms                    = Int(600)
#builder.dataset_augmentation.replicate.min_dist                 = Float(24)
#builder.dataset_augmentation.replicate.max_atoms                = Int(600)
#builder.dataset_augmentation.vacuum                             = Float(10.) # Need for isolated atoms. Should be > the cutoff
#builder.dataset_augmentation.substitution.switches_fraction     = Float(0.2)
#builder.dataset_augmentation.substitution.structures_fraction   = Float(0.1)

###############################################
# Setup Ab initio labelling
###############################################

builder.ab_initio_labelling.group_label                                                     = Str("SiC")
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.withmpi                     = True
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.max_wallclock_seconds       = QE_time
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.import_sys_environment      = False
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.resources                   = {'num_machines': QE_machine["nodes"], 
                                                                                               'num_mpiprocs_per_machine': QE_machine["taskpn"], 
                                                                                               'num_cores_per_mpiproc': QE_machine['cpupt']}
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.account                     = QE_machine['account']
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.queue_name                  = QE_machine['partition']
builder.ab_initio_labelling.quantumespresso.pw.metadata.options.custom_scheduler_commands   = '\n'.join([
    "#SBATCH --export=NONE",
   f"#SBATCH --gres=gpu:{QE_machine['gpu']}",
    "#SBATCH --constraint=A100",
])

builder.ab_initio_labelling.quantumespresso.pw.metadata.options.prepend_text               += "\n" + f"export OMP_NUM_THREADS={QE_machine['cpupt']}\n"
#builder.ab_initio_labelling.batch_size                                                      = 10    # Limit number of submitted jobs to avoid storage/job quotas or overwhelming the daemon 
builder.ab_initio_labelling.quantumespresso.clean_workdir                                   = True  
builder.ab_initio_labelling.quantumespresso.max_iterations                                  = 3     # Max number of times we can fail SCF convergence



# Can either set kpoints to a KpointsData object or set kpoints_distance in 1/Ang
kpoints = KpointsData()
kpoints.set_kpoints_mesh([1,1,1]) # Gamma point
builder.ab_initio_labelling.quantumespresso.kpoints = kpoints

# Manual overrides
qe_parameters = builder.ab_initio_labelling.quantumespresso.pw.parameters.get_dict()
# CONTROL
qe_parameters['CONTROL']['disk_io']     = 'low'
# SYSTEM
qe_parameters['SYSTEM']['occupations']  = 'fixed'
# ELECTRONS
qe_parameters['ELECTRONS']['electron_maxstep']  = 50
qe_parameters['ELECTRONS']['mixing_beta']       = 0.4
qe_parameters['ELECTRONS']['conv_thr']          = 1e-7 

builder.ab_initio_labelling.quantumespresso.pw.parameters = Dict(qe_parameters)

##############################################
# Setup MACE
###############################################

MACE_config                                                             = os.path.join(script_dir, 'mace_config.yaml')
with open(MACE_config, 'r') as yaml_file:
    mace_config = yaml.safe_load(yaml_file)
builder.training.mace.train.mace_config                                 = Dict(mace_config)

builder.training.mace.train.code                                        = MACE_train_code
builder.training.mace.train.preprocess_code                             = MACE_preprocess_code
builder.training.mace.train.postprocess_code                            = MACE_postprocess_code
builder.training.mace.train.do_preprocess                               = Bool(False)

builder.training.num_potentials                                         = Int(2)
builder.training.mace.train.metadata.options.withmpi                    = False
builder.training.mace.train.metadata.options.resources                  = {
                                                                            'num_machines': MACE_machine['nodes'],
                                                                            'num_mpiprocs_per_machine': MACE_machine['taskpn'],
                                                                            'num_cores_per_mpiproc': MACE_machine['cpupt'],
                                                                          }
builder.training.mace.train.metadata.options.max_wallclock_seconds      = MACE_time
builder.training.mace.train.metadata.options.import_sys_environment     = False
builder.training.mace.train.metadata.options.account                    = MACE_machine['account']
builder.training.mace.train.metadata.options.queue_name                 = MACE_machine['partition']
builder.training.mace.train.metadata.options.custom_scheduler_commands  = "\n".join([
    f"#SBATCH --gres=gpu:{MACE_machine['gpu']}",
     "#SBATCH --export=NONE",
     "#SBATCH --constraint=H200"
])

# For multihead fine-tuning
builder.training.mace.train.protocol                                    = Str("naive-finetune")
builder.training.mace.train.finetune_model                              = load_node(2777) # mace-mh-0 omat-pbe head
#builder.training.mace.train.finetune_replay_dataset                     = load_node(7670) # mace-mh-1 replay La-Sr-Co-O combinations

# Setup LAMMPS
###############################################

builder.exploration.entry_point                                             = Str("trains_pot.lammpsmd")

#builder.exploration.input_dataset                                           = input_structures
builder.exploration.num_random_input_structures                             = Int(10)

# If builder.bypass_exploration = True, none of the exploration parameters below matter.
# The (randomly selected) lammps input structures are passed as-is for committee evaluation.
# This is mainly useful when using some ex-situ method of exploring structures and setting
#   that dataset to lammps_input_structures

# Generate define LAMMPS trajectory parameters 
temperatures                                                                = [500]           # Kelvin
pressures                                                                   = [0]           # bar
steps                                                                       = [1000] 
styles                                                                      = ["npt"]
timestep                                                                    = 0.001     # ps

# The namespace exploration.inputs is dynamic to allow any workchain that takes
# Input:
#   potential_ase/potential_lammps (SinglefileData)
#   input_structures (PESData)
# Output:
#   explored_dataset (PESData)
# We have to provide the other inputs as a big dumb dictionary
builder.exploration.inputs = {
    "params_list": List(generate_lammps_md_config(temperatures, pressures, steps, styles, timestep)),
    "parameters": Dict({
        "control": {
            "timestep": timestep,
        },
    }),
    "potential_pair_style": Str("symmetrix/mace"),
    "sampling_time":        Float(0.1),
    "thermalization_time":  Float(0.5),
    "md": {
        "lammps": {
            "settings": Dict({
                "additional_cmdline_params": [
                    "-k", "on", "g", "1", 
                    "-sf", "kk", 
                    "-pk", "kokkos", "newton", "on", "neigh", "half"
                ]
            }),
        },
        "options": {
            "resources": {
                'num_machines': LAMMPS_machine['nodes'],
                'num_mpiprocs_per_machine': LAMMPS_machine['taskpn'],
                'num_cores_per_mpiproc': LAMMPS_machine['cpupt'],
            },
            "max_wallclock_seconds": LAMMPS_time,
            "import_sys_environment": False,
            "account": LAMMPS_machine['account'],
            "queue_name": LAMMPS_machine['partition'],
            "custom_scheduler_commands": "\n".join([
                f"#SBATCH --gres=gpu:{LAMMPS_machine['gpu']}",
                 "#SBATCH --export=NONE",
                 "#SBATCH --constraint=A100",
            ]),
        },
    },             
}

# Setup committee Evaluation
###############################################

builder.committee_evaluation.code                                           = EVALUATION_code
builder.committee_evaluation.metadata.options.resources                     = {
    'num_machines': EVALUATION_machine['nodes'],
    'num_mpiprocs_per_machine': EVALUATION_machine['taskpn'],
    'num_cores_per_mpiproc': EVALUATION_machine['cpupt']
}
builder.committee_evaluation.metadata.options.max_wallclock_seconds         = EVALUATION_time
builder.committee_evaluation.metadata.options.import_sys_environment        = False
builder.committee_evaluation.metadata.options.queue_name                    = EVALUATION_machine['partition']
builder.committee_evaluation.metadata.options.custom_scheduler_commands     = "\n".join([f"#SBATCH --gres=gpu:{EVALUATION_machine['gpu']}\n",
                                                                                          "#SBATCH --export=NONE"])
builder.committee_evaluation.metadata.options.account                       = EVALUATION_machine['account']
builder.committee_evaluation.metadata.computer                              = load_computer('midway3')

calc = submit(builder)
print(f"Submitted calculation with PK = {calc.pk}")
