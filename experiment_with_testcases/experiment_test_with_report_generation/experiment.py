from abc import ABC, abstractmethod

import numpy as np

from experiment_with_testcases.experiment_test_with_report_generation.generate_embeddings import Embedding
from experiment_with_testcases.experiment_test_with_report_generation.generate_plot_to_one_line import OneLinerOutput
from experiment_with_testcases.experiment_test_with_report_generation.generate_sentiment import Sentiment
from experiment_with_testcases.experiment_test_with_report_generation.get_percentage_smilarity import SimilarityPercentage


class Experiment:
    def __init__(self):
        self.phases_plot1 = []
        self.phases_plot2 = []
        self.results = {}

    def addPhase(self, obj, plot_number):
        if plot_number == 1:
            self.phases_plot1.append(obj)
        elif plot_number == 2:
            self.phases_plot2.append(obj)

    def run(self):
        # Process for plot 1 and plot 2
        results_plot1 = self.process_phases(self.phases_plot1)
        self.results["plot_1"] = results_plot1["one_liner"]
        results_plot2 = self.process_phases(self.phases_plot2)
        self.results["plot_2"] = results_plot2["one_liner"]

        # Calculate Similarity Percentages
        similarity_processor = SimilarityPercentage()
        self.results['semantic_similarity'] = similarity_processor.getSemanticSimilarity(
            results_plot1['embedding'], results_plot2['embedding'])
        self.results['sentiment_similarity'] = similarity_processor.getSentimentAnalysis(
            results_plot1['sentiment'], results_plot2['sentiment'])
        self.results['overall_similarity'] = similarity_processor.getOverAllPercentage(
            self.results['semantic_similarity'], self.results['sentiment_similarity'])

    def process_phases(self, phases):
        results = {}
        one_line_output = None
        for phase in phases:
            if isinstance(phase, OneLinerOutput):
                one_line_output = phase.execute()
                results['one_liner'] = one_line_output
            elif isinstance(phase, Embedding):
                embedding_result = phase.execute(one_line_output)
                results['embedding'] = embedding_result
            elif isinstance(phase, Sentiment):
                sentiment_analysis = phase.execute(one_line_output)
                results['sentiment'] = sentiment_analysis
        return results

    def getResults(self):
        structured_data = {
            'plot_1' : self.results["plot_1"],
            'plot_2' : self.results["plot_2"],
            'semantic_similarity' : f"{round(self.results['semantic_similarity'],2)}%",
            'sentiment_similarity' : f"{round(self.results['sentiment_similarity'],2)}%",
            'overall_similarity' : f"{round(self.results['overall_similarity'],2)}%"
        }

        return structured_data


