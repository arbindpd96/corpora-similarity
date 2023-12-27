from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from experiment_with_testcases.experiment_test_with_report_generation.generate_plot_to_one_line import PhaseCommand

analyzer = SentimentIntensityAnalyzer()

class Sentiment(PhaseCommand) :
    def __init__(self, method):
        self.method = method

    def execute(self, *arg):
        return self.getSentiment(arg[0] if arg[0] else "")

    def getSentiment(self, str):
        vs = analyzer.polarity_scores(str)
        return [vs['neg'], vs['neu'], vs['pos']]