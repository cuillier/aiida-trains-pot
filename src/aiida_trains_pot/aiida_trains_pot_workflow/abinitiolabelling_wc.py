"""A workchain to label multiple structures using ab initio calculations."""

from aiida.common import AttributeDict
from aiida.engine import WorkChain, append_, calcfunction, while_
from aiida.orm import Dict, Group, Int, Str, List, StructureData, EnumData, load_group
from aiida.plugins import DataFactory, WorkflowFactory
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from aiida_quantumespresso.common.types import SpinType
from aiida_quantumespresso.calculations.functions.create_magnetic_configuration import create_magnetic_configuration

import copy
import warnings
import numpy as np

PwBaseWorkChain = WorkflowFactory("quantumespresso.pw.base")
PwConstrainedWorkChain = WorkflowFactory("trains_pot.constrained")
PESData = DataFactory("pesdata")


@calcfunction
def WriteLabelledDataset(non_labelled_structures, **labelled_data):
    """Create a labelled PESData dataset from non-labelled structures and labelled data."""
    labelled_dataset = []
    elem_charge = 1.60217653e-19
    gpa_to_eV_per_ang3 = -1 * 1.0e9 / elem_charge / 1.0e30

    for key, value in labelled_data.items():
        # Check if required data exists
        if (
            "forces" not in value["output_trajectory"].get_arraynames()
            or "stress" not in value["output_trajectory"].get_arraynames()
        ):
            continue  # Skip if 'forces' or 'stress' arrays are missing
        
        config_index = int(key.split("_")[1])
        labelled_structure = copy.deepcopy(non_labelled_structures.get_list()[config_index])

        labelled_structure["dft_energy"] = float(value["output_parameters"].dict.energy)
        labelled_structure["dft_forces"] = value["output_trajectory"].get_array("forces")[0].tolist()
        stress = value["output_trajectory"].get_array("stress")[0] * gpa_to_eV_per_ang3
        labelled_structure["dft_stress"] = stress.tolist()

        # Read magnetic moments into an (N, 3) array, regardless of the nspin setting
        N_atoms = int(value["output_parameters"].dict.number_of_atoms)
        if value["output_parameters"].dict.number_of_spin_components == 1:      # Un-polarized
            magmoms = np.zeros((N_atoms, 3))
        else:
            if value["output_parameters"].dict.number_of_spin_components == 2:  # Collinear
                # Always mark as polarized along the z-axis
                magnetization_directions = np.zeros((N_atoms, 3))
                magnetization_directions[:,2] = 1.0     
                magmoms = magnetization_directions * value["output_trajectory"].get_array("atomic_magnetic_moments")[-1,:,np.newaxis]
            elif value["output_parameters"].dict.number_of_spin_components == 4: # Non-collinear
                # @TODO as of v5.0.0a1, aiida-quantumespresso does not parse the atomic_magnetic_moments for non-collinear calculations.
                # It is reported (for only the final step) in the xml file, and on each step in the text output file. 
                raise NotImplementedError("Parsing non-collinear atomic magnetic moments is not implemented.")
        
        labelled_structure["dft_magmom"] = magmoms.tolist()
        labelled_dataset.append(labelled_structure)

    pes_labelled_dataset = PESData(labelled_dataset)
    return pes_labelled_dataset


class AbInitioLabellingWorkChain(WorkChain):
    """A workchain to loop over structures and submit AbInitioLabellingWorkChain."""

    @classmethod
    def define(cls, spec):
        """Input and output specification."""
        super().define(spec)
        spec.input("unlabelled_dataset", valid_type=PESData, help="Structures to label.")
        spec.input("group_label", valid_type=Str, help="Label for group.", required=False)
        spec.input(
            "onsites_hubbard", 
            valid_type=List, 
            help="List of keyword arguments for HubbardStructureData.initialize_onsites_hubbard(). One Dict per atomic species.",
            required=False,
            default=lambda: List([])
        )
        spec.input(
            "intersites_hubbard",
            valid_type=List,
            help="List of keyword arguments for HubbardStructureData.initialize_intersites_hubbard(). One Dict per pair.",
            required=False,    
            default=lambda: List([])
        ) 
        spec.input(
            "spin_type",
            valid_type=EnumData,
            help="SpinType.NONE, SpinType.COLLINEAR, SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT",
            required=False,
            default=lambda: EnumData(SpinType.NONE)
        )
        spec.input(
            "batch_size",
            valid_type=Int,
            help="Number of structures to label in each batch.",
            required=False,
            default=lambda: Int(1000),
        )
        spec.input(
            "lambda_series",
            valid_type=List,
            help="If provided, perform a series of constrained magnetization calculations with the specified lambda values.",
            required=False,
            default=lambda: List([]) 
        )
        spec.input(
            'constrained_kinds', 
            valid_type=List, 
            required=False,
            default=None,
            help="If provided, only apply constraints to kinds matching the provided symbols (e.g., `Co`)"
        )

        spec.inputs.validator = cls.validate_inputs

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="quantumespresso",
            exclude=("pw.structure",),
            namespace_options={"validator": None},
        )
        spec.output(
            "ab_initio_labelling_data",
            valid_type=PESData,
        )
        spec.outline(
            cls.setup,
            while_(cls.check_labelled)(cls.run_ab_initio_labelling),
            cls.finalize,
        )

    @classmethod
    def validate_inputs(cls, inputs, namespace):
        if inputs["spin_type"] not in [SpinType.NONE, SpinType.COLLINEAR]:
            raise ValidationError("Only SpinType.NONE and SpinType.COLLINEAR are implemented.")

    def setup(self):
        """Initialize context and input parameters."""
        # Initialize the list of structures
        self.ctx.config = 0
        self.ctx.unlabelled_structures = self.inputs.unlabelled_dataset.get_ase_list()
        self.ctx.batch_num = 0

    def check_labelled(self):
        """Check if all structures have been labelled."""
        return self.ctx.config < len(self.ctx.unlabelled_structures)

    def run_ab_initio_labelling(self):
        """Run PwBaseWorkChain for each structure."""
        # Create or load a group to track the calculations
        if hasattr(self.inputs, "group_label"):
            group_label = self.inputs.group_label.value

        else:
            group_label = f"ab_initio_labelling_{self.uuid}"
            self.report(f"Saving configurations in group {group_label}")

        try:
            group = load_group(group_label)
            self.report(f"Using existing group: {group_label}")
        except Exception:
            group = Group(label=group_label).store()
            self.report(f"Created new group: {group_label}")

        for structure in self.ctx.unlabelled_structures[
            self.ctx.config : self.ctx.config + self.inputs.batch_size.value
        ]:
            self.ctx.config += 1
            str_data = StructureData(ase=structure)            
          
            # Prepare inputs
            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace="quantumespresso"))        
            
            # Magnetic configuration
            spin_type = self.inputs.spin_type.get_member()
            if spin_type != SpinType.NONE:    
                # @TODO implement non-collinear calculations
                if spin_type == SpinType.NON_COLLINEAR:
                    raise NotImplementedError("SpinType.NON_COLLINEAR not implemented.")
                if spin_type == SpinType.SPIN_ORBIT:
                    raise NotImplementedError("SpinType.SPIN_ORBIT not implemented.")           

                # Set starting_magnetization according to start_magmom > dft_magmom > default
                if "start_magmom" in structure.arrays.keys():
                    magmom_key = "start_magmom"
                elif "dft_magmom" in structure.arrays.keys():
                    magmom_key = "dft_magmom"
                else:
                    magmom_key = None

                if magmom_key:
                    # Convert from per-atom to per-kind magnetic moments.
                    # @TODO For non-collinear moments, we need to convert from Cartesian to spherical coordinates
                    magmoms = np.array(structure.arrays[magmom_key])[:,-1].tolist()
                    magnetic_configuration = create_magnetic_configuration(
                        structure=str_data,
                        magnetic_moment_per_site=magmoms)
                    str_data = magnetic_configuration["structure"]
                    magnetic_moments = magnetic_configuration["magnetic_moments"]
                else:
                    # If we are missing a magmom key, use the builder defaults
                    magnetic_moments = None
                
                # Use the builder to generate magnetization input parameters
                # Provide pseudos and dummy cutoffs to avoid looking for a default pseudo family that might not be installed
                # @TODO find a cleaner way to implement this that doesn't rely on aiida-quantumespresso 
                overrides = {
                    "pw": {
                        "pseudos": {kind.name: inputs.pw.pseudos[kind.symbol] for kind in str_data.kinds},
                        "parameters": {
                            "SYSTEM": {
                                "ecutwfc": 60,
                                "ecutrho": 240,
                            }
                        }
                    }
                }
                magnetic_builder = PwBaseWorkChain.get_builder_from_protocol(
                    code=inputs.pw.code,
                    structure=str_data,
                    overrides=overrides,
                    spin_type=spin_type,
                    initial_magnetic_moments=magnetic_moments 
                )

                # Get the parameters determined by the atomic magnetic moments and the spin type
                magnetic_inputs = {
                    "SYSTEM": {k:v for k,v in magnetic_builder.pw.parameters.get_dict()["SYSTEM"].items()
                               if k in ["starting_magnetization", "nspin", "angle1", "angle2", "noncolin", "lspinorb"]}
                }

                # If using constrained magnetization, starting_magnetization must be in bohr magnetons, not normalized by the PP valence.                
                if "constrained_magnetization" in inputs.pw.parameters.get_dict()["SYSTEM"].keys():
                    for kind_index,kind in enumerate(str_data.kinds):
                        # Confirm the moments are normalized
                        if np.abs(magnetic_inputs["SYSTEM"]["starting_magnetization"][kind.name] >= 1):
                            continue
                        z_valence = inputs.pw.pseudos[kind.symbol].z_valence
                        magnetic_inputs["SYSTEM"]["starting_magnetization"][kind.name] *= z_valence

                inputs.pw.parameters = Dict(recursive_merge(inputs.pw.parameters.get_dict(), magnetic_inputs))
                
            # Hubbard parameters
            if self.inputs.onsites_hubbard or self.inputs.intersites_hubbard:
                str_data = HubbardStructureData.from_structure(str_data) 
                # Set onsite Hubbard parameters
                for onsite_kwargs in self.inputs.onsites_hubbard:
                    # If the calculation is spin-polarized, kinds will be renamed to accommodate different starting_magnetizations.
                    # So force use_kinds = False to define per-element (instead of explicit per-kind) Hubbard parameters.
                    if spin_type != SpinType.NONE:
                        onsite_kwargs["use_kinds"] = False
                    str_data.initialize_onsites_hubbard(**onsite_kwargs)
                # Set intersite Hubbard parameters
                for intersite_kwargs in self.inputs.intersites_hubbard:
                    if spin_type != SpinType.NONE:
                        intersite_kwargs["use_kinds"] = False
                    str_data.initialize_intersites_hubbard(**intersite_kwargs)
            
            inputs.pw.structure = str_data
            inputs.metadata.call_link_label = f"ab_initio_labelling_config_{self.ctx.config}"
            
            inputs.pw.pseudos = {kind.name: inputs.pw.pseudos[kind.symbol] for kind in str_data.kinds}
            """
            atm_types = list(str_data.get_symbols_set())
            pseudos = inputs.pw.pseudos
            inputs.pw.pseudos = {}
            for tp in atm_types:
                if tp in pseudos.keys():
                    inputs.pw.pseudos[tp] = pseudos[tp]
                else:
                    raise ValueError(f"Pseudopotential for {tp} not found")
            """

            default_inputs = {"CONTROL": {"calculation": "scf", "tstress": True, "tprnfor": True}}
            inputs.pw.parameters = Dict(recursive_merge(default_inputs, inputs.pw.parameters.get_dict()))
            
            # Submit a constrained magnetization work chain
            if self.inputs.lambda_series:
                constrained_inputs = AttributeDict()
                if "clean_workdir" in inputs:
                    constrained_inputs.clean_workdir = inputs.pop('clean_workdir')
                constrained_inputs.quantumespresso = inputs
                constrained_inputs.lambda_series = self.inputs.lambda_series
                constrained_inputs.constrained_kinds = self.inputs.constrained_kinds
                constrained_inputs = prepare_process_inputs(PwConstrainedWorkChain, constrained_inputs)
                future = self.submit(PwConstrainedWorkChain, **constrained_inputs)
                self.report(f"Launched PwConstrainedWorkChain for configuration {self.ctx.config} <{future.pk}>")
            # Submit a normal pw.x work chain
            else:  
                inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
                future = self.submit(PwBaseWorkChain, **inputs)
                self.report(f"Launched PwBaseWorkChain for configuration {self.ctx.config} <{future.pk}>")

            # Add the calculation to the group
            group.add_nodes(future)
            self.to_context(ab_initio_labelling_calculations=append_(future))

        self.ctx.batch_num += 1

    def finalize(self):
        """Finalize the workchain and collect labelled data."""
        ab_initio_labelling_data = {}
        for ii, calc in enumerate(self.ctx.ab_initio_labelling_calculations):
            if calc.exit_status > 0:
                continue
            
            # From a PwConstrainedWorkChain or other workchain that calls multiple PwBaseWorkChains
            if 'converged_workchains' in calc.outputs:  
                for label, calc_outputs in calc.outputs.converged_workchains.items():
                    ab_initio_labelling_data[f"abinitiolabelling_{ii}_{label}"] = {
                        "output_parameters": calc_outputs.output_parameters,
                        "output_trajectory": calc_outputs.output_trajectory,
                    }
            else:   # Normal PwBaseWorkChain
                ab_initio_labelling_data[f"abinitiolabelling_{ii}"] = {
                    "output_parameters": calc.outputs.output_parameters,
                    "output_trajectory": calc.outputs.output_trajectory,
                }

        pes_dataset_out = WriteLabelledDataset(
            non_labelled_structures=self.inputs.unlabelled_dataset,
            **ab_initio_labelling_data,
        )

        self.out("ab_initio_labelling_data", pes_dataset_out)
