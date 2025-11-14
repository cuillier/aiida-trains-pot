"""A workchain to label multiple structures using ab initio calculations."""

from aiida.common import AttributeDict
from aiida.engine import WorkChain, append_, calcfunction, while_
from aiida.orm import Dict, Group, Int, Str, List, StructureData, load_group
from aiida.plugins import DataFactory, WorkflowFactory
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from aiida_quantumespresso.common.types import SpinType
from aiida_quantumespresso.calculations.functions.create_magnetic_configuration import create_magnetic_configuration

import numpy as np

PwBaseWorkChain = WorkflowFactory("quantumespresso.pw.base")
PESData = DataFactory("pesdata")


@calcfunction
def WriteLabelledDataset(non_labelled_structures, **labelled_data):
    """Create a labelled PESData dataset from non-labelled structures and labelled data."""
    labelled_dataset = []
    elem_charge = 1.60217653e-19
    gpa_to_eV_per_ang3 = -1 * 1.0e9 / elem_charge / 1.0e30
    non_labbeled_list = non_labelled_structures.get_list()

    for key, value in labelled_data.items():
        # Check if required data exists
        if (
            "forces" not in value["output_trajectory"].get_arraynames()
            or "stress" not in value["output_trajectory"].get_arraynames()
        ):
            continue  # Skip if 'forces' or 'stress' arrays are missing

        labelled_dataset.append(non_labbeled_list[int(key.split("_")[1])])
        labelled_dataset[-1]["dft_energy"] = float(value["output_parameters"].dict.energy)
        labelled_dataset[-1]["dft_forces"] = value["output_trajectory"].get_array("forces")[0].tolist()
        stress = value["output_trajectory"].get_array("stress")[0] * gpa_to_eV_per_ang3
        labelled_dataset[-1]["dft_stress"] = stress.tolist()
        labelled_dataset[-1]["dft_magmom"] = value["output_trajectory"].get_array("atomic_magnetic_moments")[0].tolist()

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
            default=lambda: List([]) )
        spec.input(
            "intersites_hubbard",
            valid_type=List,
            help="List of keyword arguments for HubbardStructureData.initialize_intersites_hubbard(). One Dict per pair.",
            required=False,    
            default=lambda: List([]) ) 
        spec.input(
            "spin_type",
            valid_type=Str,
            help="SpinType.NONE, SpinType.COLLINEAR, SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT",
            required=False,
            default=lambda: Str(SpinType.NONE))
        spec.input(
            "batch_size",
            valid_type=Int,
            help="Number of structures to label in each batch.",
            required=False,
            default=lambda: Int(1000),
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
        super().validate_inputs(inputs, namespace)
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

            # Hubbard parameters
            if self.inputs.onsites_hubbard or self.inputs.intersites_hubbard:
                str_data = HubbardStructureData.from_structure(str_data) 
            for hubbard_kwargs in self.inputs.onsites_hubbard:
                str_data.initialize_onsites_hubbard(**hubbard_kwargs)
            for hubbard_kwargs in self.inputs.intersites_hubbard:
                str_data.initialize_intersites_hubbard(**hubbard_kwargs)
            
            # Prepare inputs
            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace="quantumespresso"))        
            
            # Magnetic configuration
            if self.inputs.spin_type == SpinType.COLLINEAR:
                if "start_magmom" in structure.arrays.keys():
                    # Convert from per-atom magnetic moment vectors to
                    #   per-kind magnetic moment scalars
                    magnetic_configuration = create_magnetic_configuration(
                            structure=str_data,
                            magnetic_moment_per_site=structure.arrays["start_magmom"][:,-1])
                    magnetic_structure = magnetic_configuration["structure"]
                    magnetic_moments = magnetic_configuration["magnetic_moments"]
                else:
                    # Use builder defaults
                    magnetic_structure = str_data
                    magnetic_moments = None

                magnetic_builder = PwBaseWorkChain.get_builder_from_protocol(
                        code=inputs.pw.code,
                        structure=magnetic_structure,
                        spin_type=self.inputs.spin_type,
                        initial_magnetic_moments=magnetic_moments)

                # Override the structure and magnetization-related keywords
                str_data = magnetic_builder.pw.structure
                for keyword in ["starting_magnetization", "nspin"]:
                    inputs.pw.parameters["SYSTEM"][keyword] = magnetic_builder.pw.parameters["SYSTEM"][keyword]
          
            inputs.pw.structure = str_data
            inputs.metadata.call_link_label = f"ab_initio_labelling_config_{self.ctx.config}"

            atm_types = list(str_data.get_symbols_set())
            pseudos = inputs.pw.pseudos
            inputs.pw.pseudos = {}
            for tp in atm_types:
                if tp in pseudos.keys():
                    inputs.pw.pseudos[tp] = pseudos[tp]
                else:
                    raise ValueError(f"Pseudopotential for {tp} not found")

            default_inputs = {"CONTROL": {"calculation": "scf", "tstress": True, "tprnfor": True}}
            inputs.pw.parameters = Dict(recursive_merge(default_inputs, inputs.pw.parameters.get_dict()))

            inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
            # Submit the workchain
            future = self.submit(PwBaseWorkChain, **inputs)
            self.report(f"Launched AbInitioLabellingWorkChain for configuration {self.ctx.config} <{future.pk}>")

            # Add the calculation to the group
            group.add_nodes(future)
            self.to_context(ab_initio_labelling_calculations=append_(future))

        self.ctx.batch_num += 1

    def finalize(self):
        """Finalize the workchain and collect labelled data."""
        ab_initio_labelling_data = {}
        for ii, calc in enumerate(self.ctx.ab_initio_labelling_calculations):
            if calc.exit_status == 0:
                ab_initio_labelling_data[f"abinitiolabelling_{ii}"] = {
                    "output_parameters": calc.outputs.output_parameters,
                    "output_trajectory": calc.outputs.output_trajectory,
                }

        pes_dataset_out = WriteLabelledDataset(
            non_labelled_structures=self.inputs.unlabelled_dataset,
            **ab_initio_labelling_data,
        )

        self.out("ab_initio_labelling_data", pes_dataset_out)
