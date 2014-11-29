import os
import datetime

class EvaluationWriter(object):
    target_path = None
    evaluation_framework = None
    global_csv_name = "experiments.csv"
    global_csv_header = "Optimizer,Description,NumSteps,BestResult," \
                        "TotalCost,TotalCostEval,TotalCostCore,AvgCost," \
                        "AvgCostEval,AvgCostCore\n"

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

    def write_evaluations_to_global_csv(self):
        csv_entries = self._generate_evaluation_global_csv_entries()
        csv_file = self.open_global_csv()
        csv_file.write(csv_entries)
        csv_file.close()

    def write_out_plots_all_evaluations(self):
        #take all plots from evaluation framework and write them for all evaluations
        pass

    def write_out_plots_single_evaluation(self, single_evaluation):
        #take all plots from evaluation framework and write them for all evaluations
        pass

    def _write_out_plot(self, filename, plot):
        pass

    def open_global_csv(self):
        csv_filepath = os.path.join(self.target_path, self.global_csv_name)
        file_existed = os.path.isfile(csv_filepath)

        #open for appending, create if not exists, add header
        csv_filehandle = open(csv_filepath, 'a+')
        if file_existed:
            csv_filehandle.write(self.global_csv_header)

        return csv_filehandle


    def _create_experiment_folder(self, single_evaluation):
        optimizer_name = type(single_evaluation.get('optimizer')).__name__

        #TODO change to experiment date, when we stored this
        write_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        folder_name = os.path.join(self.target_path, optimizer_name + "_" +
                                   single_evaluation.get('description', "")
                                   + "_" + write_date)

        #create experiment folder with date
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return True

        return False

    def _generate_evaluation_global_csv_entries(self, evaluation_dicts=None, delimiter=";", line_delimiter="\n"):
        if evaluation_dicts is None:
            evaluation_dicts = self.evaluation_framework.evaluations

        lines = ""
        for evaluation in evaluation_dicts:
            lines += self._generate_evaluation_global_csv_entry(evaluation, delimiter) + line_delimiter

        return lines



    def _generate_evaluation_global_csv_entry(self, evaluation_dict, delimiter=";"):
        """
        TODO DOC to be done!

        A line with Optimizer,Description,NumSteps,BestResult,TotalCost,TotalCostEval,TotalCostCore,AvgCost,AvgCostEval,AvgCostCore
        """
        toWrite = []

        #optimizer_name
        toWrite.append(type(evaluation_dict['optimizer']).__name__)
        #description
        toWrite.append(str(evaluation_dict['description']))

        num_steps = len(evaluation_dict['result_per_step'])
        toWrite.append(str(num_steps))

        #best_result
        toWrite.append(str(evaluation_dict['best_result_per_step'][-1]))

        #compute costs
        total_cost_eval = sum(evaluation_dict['cost_eval_per_step'])
        total_cost_core = sum(evaluation_dict['cost_core_per_step'])
        total_cost = total_cost_core + total_cost_eval

        #write costs
        toWrite.append(str(total_cost))
        toWrite.append(str(total_cost_eval))
        toWrite.append(str(total_cost_core))

        #average costs
        avg_cost = float(total_cost) / float(num_steps)
        avg_cost_eval = float(total_cost_eval) / float(num_steps)
        avg_cost_core = float(total_cost_core) / float(num_steps)

        toWrite.append(str(avg_cost))
        toWrite.append(str(avg_cost_eval))
        toWrite.append(str(avg_cost_core))

        csv_string = ""
        for single in toWrite:
            csv_string += str(single) + str(delimiter)

        return csv_string






