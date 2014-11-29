import os
import datetime

class EvaluationWriter(object):
    """
    Evaluation Writer
    ------------------

    TODO Description to be done.

    """
    target_path = None
    evaluation_framework = None
    global_csv_name = "experiments.csv"

    global_csv_header = "Optimizer,Description,ObjFunction,StartDate (UTC)," \
                        "EndDate (UTC),NumSteps,BestResult,TotalCost," \
                        "TotalCostEval,TotalCostCore,AvgCost,AvgCostEval," \
                        "AvgCostCore\n"

    def __init__(self, evaluation_framework, target_path=None):
        self.evaluation_framework = evaluation_framework

        self.target_path = target_path
        if self.target_path is None:
            #look up target path from environment variable
            self.target_path = os.environ.get('APSIS_CSV_TARGET_FOLDER', None)
            if self.target_path is None:
                raise ValueError("CSVWriter needs either to be given the"
                                 "target directory to write to or the "
                                 "environment variable APSIS_CSV_TARGET_FOLDER"
                                 "must be set.")



    ##############
    # Methods for global csvs
    ##############

    def write_evaluations_to_global_csv(self):
        csv_entries = self._generate_evaluation_global_csv_entries()
        csv_file = self.open_global_csv()
        csv_file.write(csv_entries)
        csv_file.close()

    def open_global_csv(self, create_path_if_not_exists=True):
        csv_filepath = os.path.join(self.target_path, self.global_csv_name)

        #check if path exists, if not create
        if create_path_if_not_exists:
            if not os.path.exists(self.target_path):
                os.makedirs(self.target_path)

        file_existed = os.path.isfile(csv_filepath)

        #open for appending, create if not exists, add header
        csv_filehandle = open(csv_filepath, 'a+')

        #only add header if not existed
        if not file_existed:
            csv_filehandle.write(self.global_csv_header)

        return csv_filehandle

    def _generate_evaluation_global_csv_entries(self, evaluation_dicts=None, delimiter=";", line_delimiter="\n"):
        """
        Generate a string representing the bookeeping for the given single
        evaluation in the global experiments csv file.
        This method will not write to any file but just generate the
        entry.

        Parameters
        ---------
        evaluation_dicts: list of dicts
            A list of dicts corresponding to the one in EvaluationFramework.evaluations
            that contains bookeeping info of evaluations.

        delimiter: String
            Delimiter to be used for csv writing between attributes.
            Default: ";"

        line_delimiter: String
            Delimiter to be used to terminate lines. Default: "\n"

        Returns
        -------

        A string containing the lines created by
        self._generate_evaluation_global_csv_entry delimited by line_delimiter.
        """
        if evaluation_dicts is None:
            evaluation_dicts = self.evaluation_framework.evaluations

        lines = ""
        for evaluation in evaluation_dicts:
            lines += self._generate_evaluation_global_csv_entry(evaluation, delimiter) + line_delimiter

        return lines



    def _generate_evaluation_global_csv_entry(self, single_evaluation, delimiter=";"):
        """
        Generate a string representing the bookeeping for the given single
        evaluation in the global experiments csv file.
        This method will not write to any file but just generate the
        entry.

        Parameters
        ---------
        single_evaluation: dict
            A dict corresponding to the format in EvaluationFramework.evaluations
            that contains bookeeping info on this evaluation.

        delimiter: String
            Delimiter to be used for csv writing. Default: ";"

        Returns
        -------

        A one line string with Optimizer,Description,ObjFunction,StartDate,\
        EndDate,NumSteps,BestResult,TotalCost,TotalCostEval,TotalCostCore,\
        AvgCost,AvgCostEval,AvgCostCore
        """
        to_write = []

        #optimizer_name
        to_write.append(type(single_evaluation['optimizer']).__name__)
        #description
        to_write.append(str(single_evaluation['description']))

        #obj_func_name
        to_write.append(str(single_evaluation['objective_function']))

        #start date
        start_date = datetime.datetime.utcfromtimestamp(single_evaluation['start_date']).strftime("%Y-%m-%d_%H:%M:%S")
        to_write.append(str(start_date))

        #end date if exists
        end_date="None"
        if single_evaluation['end_date'] is not None:
            end_date = datetime.datetime.utcfromtimestamp(single_evaluation['end_date']).strftime("%Y-%m-%d_%H:%M:%S")
        to_write.append(str(end_date))

        num_steps = len(single_evaluation['result_per_step'])
        to_write.append(str(num_steps))

        #best_result
        to_write.append(str(single_evaluation['best_result_per_step'][-1]))

        #compute costs
        total_cost_eval = sum(single_evaluation['cost_eval_per_step'])
        total_cost_core = sum(single_evaluation['cost_core_per_step'])
        total_cost = total_cost_core + total_cost_eval

        #write costs
        to_write.append(str(total_cost))
        to_write.append(str(total_cost_eval))
        to_write.append(str(total_cost_core))

        #average costs
        avg_cost = float(total_cost) / float(num_steps)
        avg_cost_eval = float(total_cost_eval) / float(num_steps)
        avg_cost_core = float(total_cost_core) / float(num_steps)

        to_write.append(str(avg_cost))
        to_write.append(str(avg_cost_eval))
        to_write.append(str(avg_cost_core))

        csv_string = ""
        for single in to_write:
            csv_string += str(single) + str(delimiter)

        return csv_string

    ##############
    # Methods for detailed csvs
    ##############

    def append_evaluations_to_detailed_csv(self):
        csv_entries = self._generate_evaluation_global_csv_entries()
        csv_file = self.open_global_csv()
        csv_file.write(csv_entries)
        csv_file.close()

    def _create_experiment_folder(self, single_evaluation):
        """
        Create the experiment folder for a single evaluation run. The folder
        will be named

            optimizer_name_obj_func_description_start_date

        Parameters
        ---------
        single_evaluation: dict
            A dict corresponding to the format in EvaluationFramework.evaluations
            that contains bookeeping info on this evaluation.

        """
        optimizer_name = type(single_evaluation.get('optimizer')).__name__

        #read up experiment data and write it - for now use start time
        write_date = datetime.datetime.utcfromtimestamp(single_evaluation['start_date']).strftime("%Y-%m-%d_%H:%M:%S")
        folder_name = os.path.join(self.target_path, optimizer_name + "_" +
                                   single_evaluation.get('description', "")
                                   + "_" + single_evaluation.get('objective_function', "")
                                   + "_" + write_date)

        #create experiment folder with date
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return True

        return False

    ##################
    # Plots
    ##################

    def write_out_plots_all_evaluations(self):
        #take all plots from evaluation framework and write them for all evaluations
        pass

    def write_out_plots_single_evaluation(self, single_evaluation):
        #take all plots from evaluation framework and write them for all evaluations
        pass

    def _write_out_plot(self, filename, plot):
        pass








