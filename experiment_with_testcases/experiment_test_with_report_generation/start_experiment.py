from experiment_with_testcases.experiment_test_with_report_generation.experiment import Experiment
from experiment_with_testcases.experiment_test_with_report_generation.generate_embeddings import Embedding
from experiment_with_testcases.experiment_test_with_report_generation.experiment_manager import ExperimentManager
from experiment_with_testcases.experiment_test_with_report_generation.generate_plot_to_one_line import OneLinerOutput
from experiment_with_testcases.experiment_test_with_report_generation.generate_sentiment import Sentiment

from collections import OrderedDict


def insert_at_top(original_dict, key, value):
    return OrderedDict([(key, value)] + list(original_dict.items()))

def extract_id_and_type(plot_id):
    """
    Extracts the number and type (fake or real) from a given plot_id string.

    :param plot_id: A string in the format 'fake#number' or 'real#number'
    :return: A tuple containing the extracted number and type
    """
    plot_type, plot_number = plot_id.split('#')

    if plot_type == 'real':
        return int(plot_number), 'real'
    elif plot_type == 'scripted':
        return int(plot_number), 'scripted'
    else:
        return plot_id, ""


def run_experiment(config):
    manager = ExperimentManager()
    manager.load_plots_from_csv(csv_file='../../data/currated_dataset/real_movie_plots.csv' , type = 'real')
    manager.load_plots_from_csv(csv_file='../../data/test_dataset/test_movie_plots.csv' , type = 'scripted')
    type1 = None
    type2 = None

    # Iterate over each experiment in the config
    for experiment_name, settings in config.items():
        plot1_id, type1 = extract_id_and_type(settings['plot_1_id'])
        plot2_id, type2 = extract_id_and_type(settings['plot_2_id'])
        plot1 = manager.get_plot_by_id(plot_id=plot1_id, type= type1)
        plot2 = manager.get_plot_by_id(plot_id=plot2_id, type= type2)

        # Define the phases for each plot
        phases_plot1 = [
            OneLinerOutput(temp=settings['temp'], plot=plot1,one_liner_prompt= settings['one_liner_prompt']),
            Embedding(method=settings['embedding_method'], plot=plot1),
            Sentiment(settings.get('sentiment_method', ''))
        ]
        phases_plot2 = [
            OneLinerOutput(temp=settings['temp'], plot=plot2, one_liner_prompt=  settings['one_liner_prompt']),
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
        results[experiment_name] = insert_at_top(results[experiment_name], 'embedding_method',
                                                 config[experiment_name]['embedding_method'])
        results[experiment_name] = insert_at_top(results[experiment_name], 'temp', config[experiment_name]['temp'])
        results[experiment_name] = insert_at_top(results[experiment_name], 'one_liner_prompt', config[experiment_name]['one_liner_prompt'])
        results[experiment_name]['desired_similarity'] = config[experiment_name]['desired_similarity']
        results[experiment_name]['result_note'] = config[experiment_name]['result_note']


    # Create and print comparison table
    table = manager.create_comparison_table(results, experiment_folder_name, type1, type2)
    print(table)


def Merge(dict1, dict2):
    dict2.update(dict1)
    return dict2

testCases = [{
    'plot_1_id': 'real#1',
    'plot_2_id': 'scripted#6',
    'desired_similarity': 'High',
    'result_note' : 'both the movie is similar on the basis of genre and the movie narration and entire execution of movie is very similar'
}, {
    'plot_1_id': 'real#1',
    'plot_2_id': 'scripted#7',
    'desired_similarity': 'Medium',
    'result_note' : 'Both the movies have some level of similarity but not completely similar'
},
    {
    'plot_1_id': 'scripted#4',
    'plot_2_id': 'scripted#5',
    'desired_similarity': 'Low',
    'result_note' : 'Both the movies are very different as one is a mental health related movie with suspense and 2nd movie is a romantic love story'
}
]

experiment_group = {
    "name": "temp change",
    "experiments": {
        'Experiment_1': {
            'temp': 0.7,
            'one_liner_prompt': "Condense the entire plot of this movie into a single, comprehensive sentence under 50 words, highlighting the crucial events, character relationships, and the overarching theme that defines the story.",
            'embedding_method': 'openai',
            'sentiment_method': '',
        },
        'Experiment_2': {
            'temp': 0.7,
            'one_liner_prompt': "Summarize the main storyline of this movie in one line, combining key plot points, character arcs, and the core message into a brief yet descriptive narrative of less than 50 words.",
            'embedding_method': 'sbert',
            'sentiment_method': '',

        },
        'Experiment_3': {
            'temp': 0.7,
            'one_liner_prompt' : "Craft a concise, one-sentence overview of this movie's plot, seamlessly integrating major twists, character developments, and the final resolution in a summary not exceeding 50 words.",
            'embedding_method': 'bert',
            'sentiment_method': '',
        },
        'Experiment_4': {
            'temp': 0.7,
            'one_liner_prompt' : "",
            'embedding_method': 'openai',
            'sentiment_method': ''
        }
    }
}


# def main():
#     for i in testCases:
#         abc = {}
#         for (exp, cc) in experiment_group.get("experiments").items():
#             abc[exp] = Merge(cc, i)
#         run_experiment(abc)

def main():
    for test_case in testCases:
        merged_results = {}
        for experiment_name, experiment_details in experiment_group["experiments"].items():
            merged_entry = {**test_case, **experiment_details}
            merged_results[experiment_name] = merged_entry
        run_experiment(merged_results)


experiment_folder_name = "Experiment 3"
main()
