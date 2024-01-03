import csv
import os
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from experiment_with_testcases.experiment_test_with_report_generation.generate_embeddings import Embedding
from experiment_with_testcases.experiment_test_with_report_generation.experiment import Experiment
from prettytable import PrettyTable, ALL
from PIL import Image, ImageDraw, ImageFont
import textwrap


class ExperimentManager:
    def __init__(self):
        self.experiments = {}
        self.results = {}
        self.real_plots = {}
        self.scripted_plots = {}

    def add_experiment(self, name, phases_plot1, phases_plot2):
        experiment = Experiment()
        for phase in phases_plot1:
            experiment.addPhase(phase, plot_number=1)
        for phase in phases_plot2:
            experiment.addPhase(phase, plot_number=2)
        self.experiments[name] = experiment

    def run_all_experiments(self):
        for name, experiment in self.experiments.items():
            experiment.run()
            result_plot = experiment.getResults()
            self.results[name] = result_plot

    def get_results(self):
        return self.results

    def wrap_text(self, text, width=30):
        """Wraps text to the specified width."""
        return '\n'.join(textwrap.wrap(text, width))

    def create_comparison_table(self, data, folderName, *args):
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        # Initialize an ordered list for row labels
        row_labels = []
        for experiment in data.values():
            for key in experiment.keys():
                if key not in row_labels:
                    row_labels.append(key)

        field_names = ['Metrics'] + list(data.keys())

        # Create and format the table
        table = PrettyTable()
        table.field_names = field_names
        table.hrules = ALL
        table.vrules = ALL
        table.align = "l"

        # Determine which fields may contain long text
        long_text_fields = { f"plot_1", f"plot_2", "result_note", "one_liner_prompt"}

        # Fill the table with data in the order of row labels
        for label in row_labels:
            if label == "plot_1":
                lbl_name = f"plot 1({args[0]})"
            elif label == "plot_2":
                lbl_name = f"plot 2({args[1]})"
            else :
                lbl_name = label
            row = [lbl_name]
            for experiment_name in field_names[1:]:  # Skip the 'Metrics' label
                content = data[experiment_name].get(label, "N/A")
                # Wrap text for specific fields
                if label in long_text_fields:
                    content = self.wrap_text(content, width=30)
                row.append(content)
            table.add_row(row)

        # Call the function to convert the table to an image
        self.table_to_image(table, folderName)

        return table

    def load_plots_from_csv(self, csv_file, type):
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for _, row in enumerate(reader):
                if type == 'real' :
                    self.real_plots[row['id']] = row['overview']
                else :
                    self.scripted_plots[row['id']] = row['overview']

    def get_plot_by_id(self, plot_id, type):
        return self.real_plots.get(str(plot_id), None) if type == 'real' else self.scripted_plots.get(str(plot_id), None)

    def table_to_image(self, table, folderName):
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # Create the folder path
        folder_path = f'../../output/{folderName}'

        # Check if the folder exists, if not create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Incorporate the formatted date and time into the filename
        image_path = f'{folder_path}/table_image_{current_datetime}.png'

        # Convert table to string and split into lines
        table_str = table.get_string()
        lines = table_str.split('\n')

        # Create a matplotlib figure
        plt.figure(figsize=(10, len(lines) * 0.5))
        plt.text(0.5, 0.5, '\n'.join(lines),
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=12, family='monospace')

        # Remove axes
        plt.axis('off')

        # Save the figure as an image
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        return image_path
