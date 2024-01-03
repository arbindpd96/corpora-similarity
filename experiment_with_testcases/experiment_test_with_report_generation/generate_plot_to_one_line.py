from abc import abstractmethod, ABC

from prettytable import PrettyTable, ALL

from utils.infer_from_gpt import infer_gpt_4


class PhaseCommand(ABC):
    @abstractmethod
    def execute(self, *arg):
        pass


class OneLinerOutput(PhaseCommand) :
    def __init__(self, temp, plot, one_liner_prompt):
        self.temp = temp
        self.plot = plot
        self.one_liner_prompt : str = one_liner_prompt

    def execute(self, *arg):
        return self.processPlotToOneLine(plot=self.plot)

    def processPlotToOneLine(self, plot : str) :
        system_prompt = self.one_liner_prompt if self.one_liner_prompt != ""  else "Based on the detailed plot provided, synthesize a one-sentence summary that encapsulates all key events and the overall story arc. Focus on merging the main plot twists, character dynamics, and the final outcome into a cohesive and succinct narrative line. Output short be short and less then 20-30 words explain entire movie motive and flow of event."
        user_prompt = plot
        if len(user_prompt) < 500 :
            return user_prompt
        else :
            output = infer_gpt_4(system_prompt= system_prompt, user_prompt= user_prompt, temperature= self.temp)
            print(f"one liner -> {output}")
            return output


