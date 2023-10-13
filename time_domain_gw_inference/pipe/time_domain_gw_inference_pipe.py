import abc
import configparser
import argparse
import os
import shutil
import pathlib
from dataclasses import dataclass
import numpy as np
from ezdag import DAG, Layer, Node, Option
from typing import Dict, List, Tuple
from htcondor.dags import SimpleFormatter


def get_option_from_list(option_name: str, option_list: list[Option]):
    return next((opt for opt in option_list if opt.name == option_name), None)


def set_option_in_list(option_list: list[Option], new_option: Option) -> None:
    """
    If the option already exists in the list, update the argument, otherwise append it to the list
    :param option_list:
    :param new_option:
    :return:
    """
    old_option = get_option_from_list(new_option.name, option_list)
    if old_option is None:
        option_list.append(new_option)
    else:
        option_list.pop(option_list.index(old_option))
        option_list.append(new_option)
    return


@dataclass
class AbstractPipelineDAG(abc.ABC):
    output_directory: str
    config_file: str
    submit: bool

    def __post_init__(self):
        self.executables, self.condor_settings, self.time_domain_gw_inference_settings = \
            self.parse_config(self.config_file)

    def default_condor_settings(self):
        condor_settings = {
            "universe": "vanilla",
            "when_to_transfer_output": "ON_EXIT_OR_EVICT",
            "success_exit_code": 0,
            "getenv": "True",
            "log": "logs/$(nodename)-$(cluster)-$(process).log",
            "initialdir": os.path.abspath(self.output_directory),
            "notification": "ERROR",
        }
        return condor_settings

    @staticmethod
    def validate_executables(executables: Dict[str, str]):
        if not os.path.exists(executables['run_sampler']):
            raise FileNotFoundError(f"Executable {executables['run_sampler']} not found")
        executables['run_sampler'] = os.path.abspath(executables['run_sampler'])

    @staticmethod
    def validate_condor_settings(condor_settings: Dict[str, str]):
        if condor_settings.get("accounting_group_user", None) is None:
            print("WARNING: accounting_group_user not set under [condor] in the config file,"
                  " this may be a problem for some clusters")
        if condor_settings.get("accounting_group", None) is None:
            print("WARNING: accounting_group not set under [condor] in config file,"
                  " this may be a problem for some clusters")

    @staticmethod
    def validate_run_settings(run_settings: Dict[str, str]):
        return

    def parse_config(self, config_file: str) \
            -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """
        Read the configuration file and extract values from the [data], [time_domain_gw_inference],
         and [executables] sections.

        Args:
            :param config_file: (str) The path to the configuration file.
            :param output_directory: (str) The path to the output directory that contains the dag files,
                config file, and submit script, and time_domain_gw_inference output directories.
        Returns:
            tuple[dict[str, str], dict[str, str], dict[str, str]]: A tuple containing 5 dictionaries,
            the first one for [data] section values, the second one for [time_domain_gw_inference] section values,
            and the third one for [executables] section values.

        """
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.optionxform = str  # Preserve case

        config.read(config_file)

        executables = dict(config.items("executables"))

        condor_settings = self.default_condor_settings()
        condor_settings.update(dict(config.items("condor")))

        run_settings = dict(config.items("time_domain_gw_inference"))

        self.validate_executables(executables)
        self.validate_condor_settings(condor_settings)
        self.validate_run_settings(run_settings)

        return executables, condor_settings, run_settings

    def submit_dag_or_print_instructions(self, dag_file):
        if self.submit:
            print("TODO submit has not been implemented yet :-( ")
            # dag_submit = htcondor.Submit.from_dag(dag_file, {'force': 1})
        else:
            print(f"******************************************************")
            print(f"To submit the DAG, run the following command:")
            print(f"\tcondor_submit_dag -import_env -usedagdir {dag_file} ")
            print(f"******************************************************")

    @abc.abstractmethod
    def attach_layers_to_dag(self, dag):
        raise NotImplementedError("add_jobs_to_layers has not been implemented yet")

    def create_pipeline_dag(self):
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)

        # Create the DAG
        dag = DAG(formatter=SimpleFormatter())

        self.attach_layers_to_dag(dag)

        dagName = f"{os.path.basename(self.output_directory)}.dag"

        # Write the DAG to a file
        dag.write_dag(dagName, path=pathlib.Path(self.output_directory))
        dag.write_script(os.path.join(self.output_directory, "command_line.sh"))

        # Copy config file to output directory
        destination_path = os.path.join(self.output_directory, 'config.ini')
        shutil.copy(self.config_file, destination_path)

        dag_file = os.path.abspath(os.path.join(self.output_directory, dagName))

        self.submit_dag_or_print_instructions(dag_file)


@dataclass
class RunSamplerDag(AbstractPipelineDAG):
    cycle_list: List[float]
    """
    A class for creating a DAG for a single event BayesWave pipeline where the data
    does not depend on the model settings
    Will not matter if the data is real or simulated
    """

    @staticmethod
    def _copy_file_to_directory_and_return_new_name_(file, target_directory, relative_path=None):
        shutil.copy(file, target_directory)

        just_filename = os.path.basename(file)
        new_file_path = os.path.join(target_directory, just_filename)
        if relative_path is not None:
            new_file_path = os.path.relpath(new_file_path, relative_path)
        return new_file_path

    def move_input_files(self):
        """
        Move assorted input files to directory containing all the runs,
        then replace paths with paths relative to output_directory
        :return:
        """

        data_directory = os.path.join(self.output_directory, 'data_directory')
        os.makedirs(data_directory, exist_ok=True)

        data_dict = eval(self.time_domain_gw_inference_settings['data-path-dict'])
        psd_dict = eval(self.time_domain_gw_inference_settings['psd-path-dict'])
        new_data_dict = {}
        new_psd_dict = {}

        ifos = list(data_dict.keys())
        for ifo in ifos:
            new_data_dict[ifo] = self._copy_file_to_directory_and_return_new_name_(
                data_dict[ifo], data_directory, self.output_directory)
            new_psd_dict[ifo] = self._copy_file_to_directory_and_return_new_name_(
                psd_dict[ifo], data_directory, self.output_directory)

        injected_parameters = self.time_domain_gw_inference_settings.get('injected-parameters', None)
        if injected_parameters is None:
            pe_posterior_h5_file = self.time_domain_gw_inference_settings.get('pe-posterior-h5-file', None)
            if pe_posterior_h5_file is None:
                raise AssertionError(
                    'Neither injected-parameters nor pe-posterior-h5-file supplied, please include one')
            self.time_domain_gw_inference_settings['pe-posterior-h5-file'] = \
                self._copy_file_to_directory_and_return_new_name_(
                    pe_posterior_h5_file, data_directory, self.output_directory)
        else:
            pe_posterior_h5_file = self.time_domain_gw_inference_settings.get('pe-posterior-h5-file', None)
            if pe_posterior_h5_file is not None:
                raise AssertionError(
                    'both injected-parameters and pe-posterior-h5-file have been supplied, please only include one')
            self.time_domain_gw_inference_settings['injected-parameters'] = \
                self._copy_file_to_directory_and_return_new_name_(
                    injected_parameters, data_directory, self.output_directory)

        self.time_domain_gw_inference_settings['data-path-dict'] = '"' + str(new_data_dict) + '"'
        self.time_domain_gw_inference_settings['psd-path-dict'] = '"' + str(new_psd_dict) + '"'

    def attach_layers_to_dag(self, dag):
        self.move_input_files()
        runSamplerLayerManager = RunSamplerLayerManager(self.time_domain_gw_inference_settings,
                                                        self.executables['run_sampler'],
                                                        self.condor_settings)

        for cycle in self.cycle_list:
            if cycle == 0:
                run_modes = ['full', 'pre', 'post']
            else:
                run_modes = ['pre', 'post']

            for run_mode in run_modes:
                runSamplerLayerManager.add_job(run_mode, cycle)

        dag.attach(runSamplerLayerManager.layer)


@dataclass
class AbstractLayerManager(abc.ABC):
    run_settings_dict: Dict[str, str]
    executable_file: str
    shared_condor_settings: Dict[str, str]

    def __post_init__(self):
        self.layer = Layer(self.executable_file, name=self.method_name,
                           submit_description=self.condor_settings)

    def get_job_index(self):
        return len(self.layer.nodes)

    @property
    @abc.abstractmethod
    def method_name(self) -> str:
        raise NotImplementedError("method_name has not been implemented yet")

    @staticmethod
    def update_options_list(options_list: List[Option], new_options: List[Option]) -> None:
        """
        Update the options list with new options, if the option already exists in the list, update the argument
        :param options_list:
        :param new_options:
        :return:
        """
        if new_options is None:
            return
        for new_option in new_options:
            set_option_in_list(options_list, new_option)

    @property
    @abc.abstractmethod
    def condor_settings(self):
        raise NotImplementedError("condor_settings has not been implemented yet")

    def raise_option_exists_error(self, option_name, option_list) -> None:
        if get_option_from_list(option_name, option_list) is not None:
            raise ValueError(f"{option_name} option already exists in {self.method_name} settings, "
                             f"please remove it from the [{self.method_name}] section in the config file.")
        return

    @abc.abstractmethod
    def get_run_options(self, **kwargs) -> List[Option]:
        """
        Get the command line options for the executable
        :return:
        """
        raise NotImplementedError("get_run_options has not been implemented yet")

    @abc.abstractmethod
    def add_job(self, **kwargs) -> None:
        """
        Add a job to the layer
        :return:
        """
        return


@dataclass
class RunSamplerLayerManager(AbstractLayerManager):
    @property
    def method_name(self) -> str:
        return "run_sampler"

    @property
    def condor_settings(self):
        additional_settings = {
            "request_memory": "4GB",
            "request_disk": "5000MB",
            "request_cpus": "64",  # TODO allow to modify
            "when_to_transfer_output": "ON_EXIT_OR_EVICT",
        }
        additional_settings.update(self.shared_condor_settings)
        return additional_settings

    @staticmethod
    def get_output_filename_prefix(run_mode, cycles):
        return f'{run_mode}_{cycles}cycles'

    def get_run_options(self, additional_options=None, **kwargs) -> List[Option]:
        """
        Get the command line options for the run_sampler executable
        :return:
        """
        run_options = [Option(key, value) for key, value in self.run_settings_dict.items()]

        if additional_options is not None:
            self.update_options_list(run_options, additional_options)

        self.raise_option_exists_error("output-h5", run_options)
        self.raise_option_exists_error("mode", run_options)
        return run_options

    def get_inputs(self):
        # Since we have already transferred the input into the data_directory, we just need to pass data_directory
        return [Option('data-directory', 'data_directory', suppress=True)]

    def get_outputs(self, run_mode, cycles):
        dat_file = f'{self.get_output_filename_prefix(run_mode, cycles)}.dat'
        h5_file = f'{self.get_output_filename_prefix(run_mode, cycles)}.h5'
        return [Option('dat_file', dat_file, suppress=True), Option('output-h5', h5_file)]

    def add_job(self, run_mode, cycles, additional_options=None) -> None:
        run_options = self.get_run_options(additional_options)
        run_options.append(Option('mode', run_mode))
        run_options.append(Option('Tcut-cycles', cycles))

        inputs = self.get_inputs()
        outputs = self.get_outputs(run_mode, cycles)

        self.layer += Node(
            arguments=run_options,
            inputs=inputs,
            outputs=outputs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and optionally submit a Condor DAG for the "
                                                 "time_domain_gw_inference pipeline "
                                                 "pipeline.")
    parser.add_argument("--config_file", required=True, help="Path to the configuration file")
    parser.add_argument("--output_directory", required=True,
                        help="The path to the output directory that contains the dag files, config file, submit "
                             "script, and run_sampler output files")
    parser.add_argument("--cycle_list", required=True, nargs='+', type=float,
                        help="Cycles before merger to cut data at, e.g. --cycle_list -3 0 1")  # TODO describe better

    parser.add_argument("--submit", action="store_true", help="Submit the DAG to Condor (NOT IMPLEMENTED YET))")
    args = parser.parse_args()

    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Config file '{args.config_file}' not found.")

    pipeline_dag = RunSamplerDag(args.output_directory, args.config_file, args.submit,
                                 cycle_list=args.cycle_list)

    pipeline_dag.create_pipeline_dag()
