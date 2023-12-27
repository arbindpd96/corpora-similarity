from sklearn.metrics.pairwise import cosine_similarity

from experiment_with_testcases.experiment_test_with_report_generation.generate_plot_to_one_line import PhaseCommand


class SimilarityPercentage :

    def __init__(self):
        pass

    def getSemanticSimilarity(self, embedding1, embedding2):
        similarity = cosine_similarity([embedding1], [embedding2])
        return similarity[0][0] * 100

    def getSentimentAnalysis(self, sentiment1, sentiment2):
        emotionSimilarity = cosine_similarity([sentiment1], [sentiment2])[0][0] * 100
        return emotionSimilarity

    def getOverAllPercentage(self, semanticPercentage, sentimentPercentage):
        avgOutput = (semanticPercentage + sentimentPercentage) / 2
        return self.scaleTheOutput(avgOutput)

    def scaleTheOutput(self,calculatedPercentage: float):
        scaledOutput = (calculatedPercentage - 50) / 0.5
        if scaledOutput < 0:
            return 0
        return scaledOutput

