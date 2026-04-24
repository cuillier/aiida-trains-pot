"""Workchain to perform exploration MD simulations with LAMMPS."""

import os
import tempfile

from pathlib import Path

from aiida.engine import WorkChain, append_, while_, calcfunction
from aiida.orm import Dict, Bool, Float, List, SinglefileData, Str, StructureData
from aiida.plugins import DataFactory, WorkflowFactory
from aiida_lammps.data.potential import LammpsPotentialData
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

from aiida_trains_pot.utils.lammps_pair_coeffs import get_dftd2_pair_coeffs, get_mace_pair_coeff, get_meta_pair_coeff
from aiida_trains_pot.utils.generate_config import generate_lammps_md_config

LammpsWorkChain = WorkflowFactory("lammps.base")
PESData = DataFactory("pesdata")

@calcfunction
def LammpsFrameExtraction(
    sampling_time,
    saving_frequency,
    thermalization_time,
    **trajectories,
):
    """Extract frames from trajectory."""
    extracted_frames = []
    for _, trajectory in trajectories.items():
        try:
            calculation = next(
                calc.node
                for calc in trajectory.base.links.get_incoming().all()
                if calc.node.process_type == "aiida.calculations:lammps.base"
            )
        except StopIteration:
            # in principle should not happen, but in case skip this trajectory
            continue
        params = calculation.inputs.parameters
        input_structure = calculation.inputs.structure

        timestep = params["control"]["timestep"]
        integration_style = params["md"]["integration"]["style"].lower()
        temperature = params["md"]["integration"]["constraints"]["temp"]

        i = int(thermalization_time.value / timestep / saving_frequency.value) if thermalization_time.value > 0 else 1

        while i < trajectory.number_steps:
            frame = trajectory.get_step_structure(i).get_ase()

            extracted_frames.append(
                {
                    "cell": frame.get_cell(),
                    "symbols": frame.get_chemical_symbols(),
                    "positions": frame.get_positions(),
                    "input_structure_uuid": str(input_structure.uuid),
                    "gen_method": "LAMMPS",
                    "pbc": frame.get_pbc(),
                }
            )
            
            extracted_frames[-1]["style"] = integration_style
            extracted_frames[-1]["temp"] = temperature
            extracted_frames[-1]["timestep"] = timestep
            extracted_frames[-1]["id_lammps"] = calculation.uuid

            i = i + int(sampling_time.value / timestep / saving_frequency.value)

    pes_extracted_frames = PESData(extracted_frames)
    return {"explored_dataset": pes_extracted_frames}


def generate_potential(potential, pair_style) -> LammpsPotentialData:
    """Generate the potential to be used in the calculation.

    Takes a potential form OpenKIM and stores it as a LammpsPotentialData object.

    :return: potential to do the calculation
    :rtype: LammpsPotentialData
    """
    potential_parameters = {
        "species": [],
        "atom_style": "atomic",
        "units": "metal",
        "extra_tags": {},
    }

    # Assuming you have a trained MACE model
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        with potential.open(mode="rb") as potential_handle:
            potential_content = potential_handle.read()
        tmp_file.write(potential_content)
        tmp_file_path = tmp_file.name

    potential = LammpsPotentialData.get_or_create(
        source=Path(tmp_file_path),
        pair_style=pair_style,
        **potential_parameters,
    )

    os.remove(tmp_file_path)

    return potential


###################################################################
##                       DEFAULT VALUES                          ##
###################################################################

DEFAULT_thermalization_time = Float(0.0)
DEFAULT_sampling_time = Float(1.0)

DEFAULT_potential_pair_style = Str("mace no_domain_decomposition")
DEFAULT_max_restarts = 3
DEFAULT_settings = Dict(
    {
        "store_restart": True,
    },
)
DEFAULT_parameters = Dict(
    {
        "structure": {
            "atom_style": "atomic",
            "atom_modify": "map yes",
            "dimension": "3",
            "boundary": "p p p",
        },
        "potential": {
            "potential_style_options": [],
        },
        "control": {
            "timestep": 0.001,
            "newton": "on",
            "units": "metal",
        },
        "thermo": {
            "printing_rate": 100,
            "thermo_printing": {
                "step": True,
                "time": True,
                "pe": True,
                "ke": True,
                "etotal": True,
                "press": True,
                "pxx": True,
                "pyy": True,
                "pzz": True,
                "temp": True,
            },
        },
        "restart": {
            "print_final": True,
        },
        "md": {},
        "dump": {},
    }
)
DEFAULT_params_list = List(
    generate_lammps_md_config(
        temperatures = [0.0],
        pressures    = [0.0],
        steps        = [1000],
        styles       = ['nve'],
        dt           = 0.001,
    )
)
###################################################################


class LammpsMDWorkChain(WorkChain):
    """A workchain to loop over structures and submit LammpsWorkChain with retries."""

    @classmethod
    def define(cls, spec):
        """Input and output specification."""
        super().define(spec)
        spec.input(
            "input_structures",
            valid_type=PESData,
            help="Input structures for lammps",
        )
        spec.input(
            "params_list", 
            valid_type=List,
            default=lambda: DEFAULT_params_list,
            help="List of ensemble parameters (temperature, pressure, etc.) for MD",
            required=False,
        )
        spec.input(
            "parameters",  
            valid_type=Dict, 
            default=lambda: DEFAULT_parameters,
            help="Global parameters for LAMMPS",
            required=False,
        )
        spec.input(
            "potential_lammps",
            valid_type=SinglefileData,
            required=False,
            help="One of the potential for MD",
        )
        spec.input(
            "potential_pair_style",
            valid_type=Str,
            default=lambda: DEFAULT_potential_pair_style,
            required=False,
            help=f"General potential pair style. Default: {DEFAULT_potential_pair_style}",
        )
        spec.input(
            "sampling_time",
            valid_type=Float,
            default=lambda: DEFAULT_sampling_time,
            required=False,
            help="Correlation time for frame extraction",
        )
        spec.input(
            "thermalization_time",
            valid_type=Float,
            default=lambda: DEFAULT_thermalization_time,
            required=False,
            help="Thermalization time to wait before sampling trajectories."
        )
        spec.input(
            "protocol",
            valid_type=Str,
            default=None,
            help="Protocol for the calculation",
            required=False,
        )

        spec.expose_inputs(
            LammpsWorkChain,
            namespace="md",
            exclude=("lammps.structure", "lammps.potential", "lammps.parameters"),
            namespace_options={"validator": None},
        )

        spec.inputs.validator = cls.validate_inputs

        spec.output(
            "explored_dataset", 
            valid_type=PESData,
            help="Exploration outputs"
        )

        spec.outline(
            cls.run_md,
            while_(cls.not_converged)(
                cls.run_restart,
            ),
            cls.finalize_md,
        )

    @classmethod
    def validate_inputs(cls, inputs, _):
        """Validate the top-level inputs."""
        protocol = inputs.get("protocol", None)
        pair_style = inputs.get("potential_pair_style", None)

        if (
            protocol is not None
            and protocol.value == "vdw_d2"
            and pair_style is not None
            and "metatomic" in pair_style.value
        ):
            return "The 'vdw_d2' protocol is not compatible with the 'metatomic' potential pair style."

    def run_md(self):  # noqa: PLR0912
        """Run MD simulations for each structure and MD parameter set, with retries on failure."""
        potential = self.inputs.potential_lammps
        self.ctx.rerun_wc = []
        self.ctx.rerun_wc_old = []
        self.ctx.last_wc = []
        self.ctx.dict_wc = {}
        self.ctx.iteration = 0
        # Loop over structures
        for structure in self.inputs.input_structures.get_ase_list():
            inputs = self.exposed_inputs(LammpsWorkChain, namespace="md")
            inputs.lammps.structure = StructureData(ase=structure)
            inputs.lammps.potential = generate_potential(potential, str(self.inputs.potential_pair_style.value))
 
            params_list = self.inputs.params_list.get_list()
            input_parameters = self.inputs.parameters.get_dict()
            
            # Set default potential information
            input_parameters.setdefault("potential", DEFAULT_parameters.get_dict()["potential"])
            
            generate_pair_coeff = "pair_coeff_list" not in input_parameters["potential"]

            # Pair coefficients for MACE potential without hybrid/overlay is always generated,
            # if needed it is overwritten
            if generate_pair_coeff:
                if "metatomic" in self.inputs.potential_pair_style.value:
                    pair_coeffs = [get_meta_pair_coeff(inputs.lammps.structure, hybrid=False)]
                else:
                    pair_coeffs = [get_mace_pair_coeff(inputs.lammps.structure, hybrid=False)]


            if "metatomic" in self.inputs.potential_pair_style.value:
                if "potential_style_options" not in self.inputs.parameters["potential"]:
                    input_parameters["potential"]["potential_style_options"] = ["potential.dat"]
            if self.inputs.protocol is not None:
                if self.inputs.protocol == "vdw_d2":
                    if "potential_style_options" not in self.inputs.parameters["potential"]:
                        input_parameters["potential"]["potential_style_options"] = [
                            "mace no_domain_decomposition momb 20.0 0.75 20.0"
                        ]
                    if generate_pair_coeff:
                        # Generate DFT-D2 pair coefficients, it overwrites the MACE pair_coeff generated above
                        pair_coeffs = get_dftd2_pair_coeffs(inputs.lammps.structure)
                        pair_coeffs.append(get_mace_pair_coeff(inputs.lammps.structure, hybrid=True))
            input_parameters["potential"]["pair_coeff_list"] = pair_coeffs

            parameters = recursive_merge(DEFAULT_parameters.get_dict(), input_parameters)
            lammps_inputs = self.inputs.md.lammps
            if "settings" in lammps_inputs:
                inputs.lammps.settings = recursive_merge(
                    DEFAULT_settings.get_dict(), self.inputs.md.lammps.settings.get_dict()
                )
            else:
                inputs.lammps.settings = DEFAULT_settings.get_dict()
            # if 'dump' not in parameters:
            #     parameters['dump'] = {}
            parameters["dump"]["dump_rate"] = int(self.inputs.sampling_time / parameters["control"]["timestep"])

            # Loop over the MD parameter sets
            for params_md in params_list:
                if not any(inputs.lammps.structure.pbc):
                    params_md["integration"]["style"] = "nvt"

                constraint = params_md["integration"]["constraints"]

                # Map dimensions to constraints
                axes = ["x", "y", "z"]

                # Remove constraints for non-periodic directions
                for idx, axis in enumerate(axes):
                    if not inputs.lammps.structure.pbc[idx]:
                        constraint.pop(axis, None)  # Avoid KeyError if axis doesn't exist

                params_md["integration"]["constraints"] = constraint

                parameters["md"] = dict(params_md)

                inputs.lammps.parameters = Dict(parameters)
                future = self.submit(LammpsWorkChain, **inputs)
                self.to_context(md_wc=append_(future))
                self.ctx.dict_wc[f"{self.ctx.iteration}"] = self.ctx.iteration
                self.ctx.last_wc.append(self.ctx.iteration)
                self.ctx.iteration += 1

    def run_restart(self):
        """Restart the failed MD simulations."""
        self.ctx.last_wc = []
        for ii, calc in enumerate(self.ctx.md_wc):
            if ii in self.ctx.rerun_wc:
                incoming = calc.base.links.get_incoming().nested()

                # Build the inputs dictionary
                inputs = self.exposed_inputs(LammpsWorkChain, namespace="md")
                for key, node in incoming.items():
                    if key == "lammps":
                        inputs[key].update(node)  # Merge nested inputs

                future = self.submit(LammpsWorkChain, **inputs)
                self.to_context(md_wc=append_(future))
                self.ctx.dict_wc[f"{self.ctx.iteration}"] = self.ctx.dict_wc[f"{ii}"]
                self.ctx.last_wc.append(self.ctx.iteration)
                self.ctx.iteration += 1

    def not_converged(self):
        """Check if any calculation did not end successfully and requires a restart."""
        # Update the old list of reruns and prepare a new one
        self.ctx.rerun_wc_old.extend(self.ctx.rerun_wc)
        self.ctx.rerun_wc = []

        for ii, calc in enumerate(self.ctx.md_wc):
            if (ii in self.ctx.last_wc) and calc.exit_status != 0:
                # Count how many times the current calculation has been retried
                retry_count = sum(1 for value in self.ctx.dict_wc.values() if value == self.ctx.dict_wc[f"{ii}"])

                # Check if the calculation failed and has been retried less than 5 times
                if retry_count < DEFAULT_max_restarts and (ii not in self.ctx.rerun_wc_old):
                    self.ctx.rerun_wc.append(ii)

        return len(self.ctx.rerun_wc) > 0
    
    def finalize_md(self):
        """Run exploration frame extraction."""
        timestep = self.ctx.md_wc[-1].inputs.lammps.parameters.get_dict()['control']['timestep']
        dump_rate = int(self.inputs.sampling_time / dt)
         
        trajectories = {}
        for ii, calc in enumerate(self.ctx.md_wc):
            if calc.exit_status == 0:
                trajectories[f"md_{ii}"] = {el: calc.outputs[el] for el in calc.outputs}

        explored_dataset = LammpsFrameExtraction(
            self.inputs.sampling_time,
            dump_rate,
            thermalization_time=self.inputs.thermalization_time,
            **trajectories,
        )["explored_dataset"]

        self.out("explored_dataset", explored_dataset)

