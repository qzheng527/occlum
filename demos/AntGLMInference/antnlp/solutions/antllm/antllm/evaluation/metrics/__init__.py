from .classification import accuracy_metrics as cam
from .classification import accuracy_macro_metrics as camm
from .classification import type_accuracy_metrics as tmm
from .classification import f1_metrics as cf1
# from .classification import calibration_metrics as cm
from .generation import bleu_metrics as bleu
from .generation import exact_match_metrics as em
from .generation import f1_metrics as generation_f1
from .generation import math_metrics as mm
from .generation import rouge_metrics as rouge
from . import triviaqa_metrics as tm
from . import gender_representation_metrics as grm
# from . import toxicity_metrics as txm
# from . import toxicity_shell_metrics as tsm
from . import perplexity_metrics as pm
from . import human_evalx_metrics as hem
from . import bbh_metrics as bbm
from . import containing_metrics


METRIC_CLASS_DICT = {
    "Accuracy": cam.HuggingfaceAccuracy,
    "AccuracyMacro": camm.HuggingfaceAccuracyMacro,
    "TypeAccuracy": tmm.HuggingfaceTypeAccuracyMacro,
    "F1": cf1.HuggingfaceF1,
    # "CalibrationError": cm.ECE,
    "Robustness": None,
    "Fairness": None,
    "GenderRepresentation": grm.HelmGenderRepresentationMetrics,
    # "Toxicity": txm.HuggingfaceToxicity,
    # "Toxicity": tsm.HuggingfaceToxicity,
    "ExactMatch": em.MultiEM,
    "GenerationF1": generation_f1.HelmF1,
    "ROUGE1": rouge.HuggingfaceRouge1,
    "ROUGE2": rouge.HuggingfaceRouge2,
    "ROUGEL": rouge.HuggingfaceRougeL,
    "ROUGE1_Chinese": rouge.ChineseRouge1,
    "ROUGE2_Chinese": rouge.ChineseRouge2,
    "ROUGEL_Chinese": rouge.ChineseRougeL,
    "BLEU": bleu.HuggingfaceBLEU,
    # "Perplexity": pm.Perplexity,
    "TriviaQA": tm.TriviaQAEval,
    "GSM8k": mm.GSM8kMetric,
    "Pass@1": hem.PassAt1,
    "BBH": bbm.BBHAccuracy,
    "ContainingAccuracy": containing_metrics.ContainingAccuracy
}

if __name__ == "__main__":
    test = METRIC_CLASS_DICT["BLEU"]
    # print(test)
    # print(test.compute([[1],[2]], [[1], [2]]))
    print(METRIC_CLASS_DICT["BLEU"].compute([["明 确 告 知"], ["遇 到 其 他 人 的 干 扰"]],
                        [["明 确 告"], ["遇 到 其 他 人 的 干"]]))
