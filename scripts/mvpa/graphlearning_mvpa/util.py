from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
)
from nipype.interfaces.fsl.model import Cluster, ClusterOutputSpec


def FindNonDummyTime(regressors_file, tr):
    """Extract the time of the first non-dummy TR,
    accounting for non-steady-state effects
    >>> from util import FindNonDummyTR
    >>> non_dummy_time = FindNonDummyTR('regressors.tsv', 0.8)
    """
    import pandas as pd
    regressors = pd.read_csv(regressors_file, sep='\t')
    non_steady_cols = regressors.columns[regressors.columns.str.startswith('non_steady_state')]
    return tr * len(non_steady_cols)


class ExtractConfoundsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    motion_params = traits.Enum(
        'motion_6',
        'motion_24',
        desc='Number of motion parameters, motion_6 or motion_24',
        mandatory=True,
        default=False)


class ExtractConfoundsOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class ExtractConfounds(BaseInterface):
    """Extract fMRIPrep motion confounds
    Reformat tsv columns into a space-separated txt file
    >>> from util import ExtractConfounds
    >>> extractconfounds = ExtractConfounds(in_file='regressors.tsv', parameters='motion_6')
    >>> res = extractconfounds.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../regressors.txt'
    .. testcleanup::
    >>> os.unlink('regressors.txt')
    """

    input_spec = ExtractConfoundsInputSpec
    output_spec = ExtractConfoundsOutputSpec


    motion_params_6 = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    motion_params_24 = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
            'trans_y', 'trans_y_derivative1', 'trans_y_power2', 'trans_y_derivative1_power2',
            'trans_z', 'trans_z_derivative1', 'trans_z_power2', 'trans_z_derivative1_power2',
            'rot_x', 'rot_x_derivative1', 'rot_x_power2', 'rot_x_derivative1_power2',
            'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2',
            'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']

    def _gen_output_file_name(self):
        import os
        _, base, _ = split_filename(self.inputs.in_file)
        return os.path.abspath(base + '.txt')

    def _run_interface(self, runtime):
        import pandas as pd
        import numpy as np

        df = pd.read_csv(self.inputs.in_file, sep='\t')
        # Derivatives for the first timepoint are N/A
        df = df.fillna(0)

        # Choose which parameters to index
        if self.inputs.motion_params == 'motion_6':
            motion_parameters = self.motion_params_6
        elif self.inputs.motion_params == 'motion_24':
            motion_parameters = self.motion_params_24
        else:
            raise ValueError('Invalid parameter name')

        # Find names of:
        # Motion Outliers
        # Non-Steady State Volumes
        outliers = df.columns[df.columns.str.startswith('motion_outlier')].values
        nonsteady = df.columns[df.columns.str.startswith('non_steady_state_outlier')].values
        confound_names = np.concatenate([motion_parameters, outliers, nonsteady])
        confounds = df.loc[:, confound_names]

        # Motion and Non-Steady State outliers may overlap
        # Remove duplicates, check if they index the same volume
        # If so, keep the non-steady-state and drop the motion outlier
        for outlier in outliers:
            vol = np.where(confounds[outlier])[0][0]
            if f'non_steady_state_outlier{vol:02}' in nonsteady:
                confounds = confounds.drop(outlier, 1)

        # Save to a format that numpy can read with readtxt
        confounds.to_csv(self._gen_output_file_name(), index=False, header=False, sep=' ')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = self._gen_output_file_name()
        return outputs


class CustomClusterOutputSpec(ClusterOutputSpec):
    output_file = File(desc="command output")


class CustomCluster(Cluster):
    """
    Update to the nipype wrapper for FSL cluster
    to capture the commandline output

    e.g. cluster.output_file
    """

    output_spec = CustomClusterOutputSpec

    def _run_interface(self, runtime):
        runtime = super()._run_interface(runtime)

        output_fname = self._gen_outfile()
        with open(output_fname, 'w') as f:
            f.write(runtime.stdout)
        return runtime

    def _gen_outfile(self):
        output_fname = self._gen_fname('cluster_' + self.inputs.in_file, suffix=None, ext='.txt')
        return output_fname

    def _list_outputs(self):
        outputs = super()._list_outputs()
        output_fname = self._gen_outfile()
        outputs['output_file'] = output_fname
        return outputs
