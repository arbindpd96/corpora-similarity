from abc import abstractmethod, ABC

from prettytable import PrettyTable, ALL

from utils.infer_from_gpt import infer_gpt_4


class PhaseCommand(ABC):
    @abstractmethod
    def execute(self, *arg):
        pass


class OneLinerOutput(PhaseCommand) :
    def __init__(self, temp, plot):
        self.temp = temp
        self.plot = plot

    def execute(self, *arg):
        return self.processPlotToOneLine(plot=self.plot)

    def processPlotToOneLine(self, plot : str) :
        system_prompt = "Based on the detailed plot provided, synthesize a one-sentence summary that encapsulates all key events and the overall story arc. Focus on merging the main plot twists, character dynamics, and the final outcome into a cohesive and succinct narrative line. Output short be short and less then 20-30 words explain entire movie motive and flow of event."
        user_prompt = plot
        return infer_gpt_4(system_prompt= system_prompt, user_prompt= user_prompt, temperature= self.temp)


