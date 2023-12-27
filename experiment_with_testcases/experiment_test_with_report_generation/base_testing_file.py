from experiment_with_testcases.experiment_test_with_report_generation.experiment import Experiment
from experiment_with_testcases.experiment_test_with_report_generation.generate_embeddings import Embedding
from experiment_with_testcases.experiment_test_with_report_generation.experiment_manager import ExperimentManager
from experiment_with_testcases.experiment_test_with_report_generation.generate_plot_to_one_line import OneLinerOutput
from experiment_with_testcases.experiment_test_with_report_generation.generate_sentiment import Sentiment

from collections import OrderedDict


def insert_at_top(original_dict, key, value):
    return OrderedDict([(key, value)] + list(original_dict.items()))

def run_experiment():
    manager = ExperimentManager()
    manager.load_plots_from_csv(csv_file='../../data/test/sample_input.csv')

    # Iterate over each experiment in the config
    for experiment_name, settings in config.items():
        plot1_id = settings['plot_1_id']
        plot2_id = settings['plot_2_id']
        plot1 = manager.get_plot_by_id(plot_id=plot1_id)
        plot2 = manager.get_plot_by_id(plot_id=plot2_id)

        # Define the phases for each plot
        phases_plot1 = [
            OneLinerOutput(temp=settings['temp'], plot=plot1),
            Embedding(method=settings['embedding_method'], plot=plot1),
            Sentiment(settings.get('sentiment_method', ''))
        ]
        phases_plot2 = [
            OneLinerOutput(temp=settings['temp'], plot=plot2),
            Embedding(method=settings['embedding_method'], plot=plot2),
            Sentiment(settings.get('sentiment_method', ''))
        ]

        # Add the experiment to the manager
        manager.add_experiment(experiment_name,
                               phases_plot1,
                               phases_plot2)

    # Run all experiments
    manager.run_all_experiments()

    # Retrieve results and augment with config settings
    results = manager.get_results()

    # Reorder the results to place 'temp' and 'embedding_method' at the top
    for experiment_name in config:
        results[experiment_name] = insert_at_top(results[experiment_name], 'embedding_method', config[experiment_name]['embedding_method'])
        results[experiment_name] = insert_at_top(results[experiment_name], 'temp', config[experiment_name]['temp'])
        results[experiment_name]['desired_similarity'] = config[experiment_name]['desired_similarity']

    # Print results
    print(results)

    # Create and print comparison table
    table = manager.create_comparison_table(results)
    print(table)




config = {
    'Experiment_1': {
        'temp': 0.3,
        'embedding_method': 'openai',
        'sentiment_method': '',
        'plot_1_id' : '7',
        'plot_2_id' : '8',
        'desired_similarity' : 'High'
    },
    'Experiment_2': {
        'temp': 0.5,
        'embedding_method': 'openai',
        'sentiment_method': '',
        'plot_1_id': '7',
        'plot_2_id': '8',
        'desired_similarity': 'High'
    },
    'Experiment_3': {
        'temp': 0.7,
        'embedding_method': 'openai',
        'sentiment_method': '',
        'plot_1_id': '7',
        'plot_2_id': '8',
        'desired_similarity': 'High'
    },
    'Experiment_4': {
        'temp': 1.0,
        'embedding_method': 'openai',
        'sentiment_method': '',
        'plot_1_id': '7',
        'plot_2_id': '8',
        'desired_similarity': 'High'
    }
}

run_experiment()




