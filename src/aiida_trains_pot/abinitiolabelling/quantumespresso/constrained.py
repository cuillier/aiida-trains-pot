from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, while_, append_, ToContext
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

class PwConstrainedWorkChain(WorkChain):
    """
    WorkChain to perform constrained magnetization calculations with a user-specified series of lambda values.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        
        spec.expose_inputs(
            PwBaseWorkChain, 
            namespace='quantumespresso',
            exclude=('clean_workdir'),
            namespace_options={
                'required': True,
                'populate_defaults': False,
                'help': (
                    'Inputs shared by each `PwBaseWorkChain`.'
                ),
            },
        )
        
        spec.input(
            'lambda_series', 
            valid_type=orm.List, 
            default=lambda: orm.List([0.0, 1.0]),
            help="List of `lambda` values to attempt, in sequence."
        )
        spec.input(
            'constrained_kinds', 
            valid_type=orm.List, 
            required=False,
            default=None,
            help="If provided, only apply constraints to kinds matching the provided symbols (e.g., `Co`)"
        )
        spec.input(
            'clean_workdir',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.',
        )
        spec.input(
           'return_uncorrelated',
            valid_type = orm.Bool,
            default = lambda: orm.Bool(True)
            help = 'If `True`, return only the last converged calculation to avoid highly correlated datasets.' 
        )        

        spec.inputs.validator = cls.validate_inputs
        
        spec.output_namespace(
            'converged_workchains',
            dynamic=True,
            help='Converged `PwBaseWorkChains` with different `lambda` values.',
        ) 
     
        spec.outline(
            cls.setup,
            while_(cls.should_continue)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            cls.results,
        )

        spec.exit_code(401, 'ERROR_UNCONSTRAINED_SCF_FAILED', message='An unconstrained `PwBaseWorkChain` failed.')

    @staticmethod
    def validate_inputs(inputs, _):
        """Validate the top level namespace."""
        parameters = inputs['quantumespresso']['pw']['parameters'].get_dict()

        # The calculation must be spin polarized.
        if 'nspin' not in parameters.get('SYSTEM', {}):
            return 'The parameters in `quantumespresso.pw.parameters` do not specify the required key `SYSTEM.nspin`.'
        elif parameters['SYSTEM']['nspin'] < 2:
            return 'The parameters in `quantumespresso.pw.parameters` specify an invalid value for `SYSTEM.nspin` (nspin > 1 required)`.'
        
        # Confirm the magnetization constraints are compatible.
        if 'constrained_magnetization' not in parameters.get('SYSTEM', {}):
            return 'The parameters in `quantumespresso.pw.parameters` do not specify the required key `SYSTEM.constrained_magnetization`.'
        else:
            if 'total' in parameters['SYSTEM']['constrained_magnetization'] and 'fixed_magnetization' not in parameters.get('SYSTEM', {}):
                return '`SYSTEM.fixed_magnetization` must be specified with `constrained_magnetization = total (direction)`.'
            if 'atomic' in parameters['SYSTEM']['constrained_magnetization'] and 'starting_magnetization' not in parameters.get('SYSTEM', {}):
                return '`SYSTEM.starting_magnetization` must be specified with `constrained_magnetization = atomic (direction)`.'


    def setup(self):
        """Initialize context variables."""
        self.ctx.iteration = 0
        self.ctx.parent_folder = None

    def run_scf(self):
        """Set input parameters and submit a PwBaseWorkChain"""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='quantumespresso'))       
        parameters = inputs.pw.parameters.get_dict()

        # Set the lambda value for this calculation.
        lambda_Ry = self.inputs.lambda_series.get_list()[self.ctx.iteration]
        parameters['SYSTEM']['lambda'] = lambda_Ry

        # If constrained_types was specified to only constrain certain atomic species (e.g., only the magnetically active species),
        # remove the starting_magnetization tags for other kinds.
        if self.inputs.constrained_kinds is not None and 'starting_magnetization' in parameters.get('SYSTEM', {}):
            kinds = parameters['SYSTEM']['starting_magnetization'].keys()
            kinds_to_remove = list(kinds)
            # Kinds will have names like `Co0`, `Co1`, ... `CoN` depending on how many unique atomic magnetic moments
            #   were provided. Keep all kinds that match a symbol (e.g. `Co`)  provided in `inputs.constrained_kinds`. 
            for kind in kinds:
                for symbol in self.inputs.constrained_kinds:
                    if symbol in kind:
                        kinds_to_remove.remove(kind)
                        break
            for kind in kinds_to_remove:
                parameters['SYSTEM']['starting_magnetization'].pop(kind)            

        # Restart from the wavefunctions of a previous calculations.
        if self.ctx.parent_folder:
            inputs.pw.parent_folder = self.ctx.parent_folder
            parameters['CONTROL']['restart_mode'] = 'restart'
            parameters['ELECTRONS']['startingwfc'] = 'file'
            parameters['ELECTRONS']['startingpot'] = 'file'
            # If a restarted calculation doesn't converge within a few SCF iterations, it likely never will.
            inputs.max_iterations = 1

        inputs.pw.parameters = orm.Dict(dict=parameters)
        inputs.metadata.call_link_label += f'_lambda_{self.ctx.iteration}'

        base_wc = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'Launching PwBaseWorkChain<{base_wc.pk}> with lambda = {lambda_Ry} Ry')
        
        return ToContext(base_workchains=append_(base_wc))

    def inspect_scf(self):
        """Confirm the previous PwBaseWorkChain finished and update context for the next iteration."""
        if self.ctx.base_workchains[-1].is_finished_ok:
            # Restart the next calculation from the current wavefunctions and charge density.
            self.ctx.parent_folder = self.ctx.base_workchains[-1].outputs.remote_folder
            self.ctx.iteration += 1
        else:
            # Only raise an error if an unconstrained calculation failed.
            # Constrained calculations are expected to fail with a high enough lambda.
            if self.inputs.lambda_series.get_list()[self.ctx.iteration] == 0.0:
                return self.exit_codes.ERROR_UNCONSTRAINED_SCF_FAILED
            else:   # Skip remaining calculations and output whatever converged.
                self.ctx.iteration = len(self.inputs.lambda_series)


    def should_continue(self):
        """Check termination conditions."""
        if self.ctx.iteration >= len(self.inputs.lambda_series):
            return False
        else:
            return True

    def results(self):
        """Output the results of each completed PwBaseWorkChain."""
        out = {}
        for ii, workchain in enumerate(self.ctx.base_workchains):
            if workchain.is_finished_ok:
                out[f'lambda_{ii}'] = {name: workchain.outputs[name] for name in workchain.outputs}

        # Only return the last calculation to converge
        if self.inputs.return_uncorrelated.value and len(out) > 1:
            for key in list(out.keys())[:-1]
                out.pop(key)       
 
        self.out(f'converged_workchains', out)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # noqa: SLF001
                    cleaned_calcs.append(called_descendant.pk)
                except (OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f'cleaned remote folders of calculations: {" ".join(map(str, cleaned_calcs))}')
