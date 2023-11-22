# import numpy as np
import calibration as cal  # requires uncertainty-calibration package
from typing import List, Dict

"""
calculate estimated calibration errors
"""


class ECE():
    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        """
        extra[0]["likelihood"]: likelihood of each label. e.g.
            {"true": 0.6, "false": 0.3, "neither": 0.1}
        combined all extras likelihood together as a list: 
            [{"true": 0.6, "false": 0.3, "neither": 0.1},
            {"true": 0.3, "false": 0.6, "neither": 0.1},
            {"true": 0.6, "false": 0.3, "neither": 0.1}]
        references: ground truth label
            [["true"], ["true"], ["false"]] 
        """
        # references = [r[0] for r in references]  # change [[1],[0],[1]] to [1, 0, 1]
        if "likelihood" not in extras[0].keys():
            raise Exception("Likelihood field is required for calibration error calculation!")
        # values_list = list(set(extras[0]["likelihood"].keys()))
        # id_dict = {value: idx for idx, value in enumerate(values_list)}
        # references = [id_dict[value] for value in references]
        likelihoods = []
        references_id = []
        max_option_length = max([len(extra["likelihood"]) for extra in extras])
        for extra, reference in zip(extras, references):
            likelihood = [0.0] * max_option_length
            for i, (k, v) in enumerate(extra["likelihood"].items()):
                likelihood[i] = v
                if k in reference:
                    references_id.append(i)
            likelihoods.append(likelihood)
        calibration_error = cal.get_calibration_error(likelihoods, references_id)
        ece = cal.get_ece(likelihoods, references_id)

        results = {"calibration error": round(100 * calibration_error, 2), 
                   "estimated calibration error": round(100 * ece, 2)}
        return results
    

# def ece_score(py, y_test, n_bins=10):
#     py = np.array(py)
#     y_test = np.array(y_test)
#     if y_test.ndim > 1:
#         y_test = np.argmax(y_test, axis=1)
#     py_index = np.argmax(py, axis=1)
#     py_value = []
#     for i in range(py.shape[0]):
#         py_value.append(py[i, py_index[i]])
#     py_value = np.array(py_value)
#     acc, conf = np.zeros(n_bins), np.zeros(n_bins)
#     Bm = np.zeros(n_bins)
#     for m in range(n_bins):
#         a, b = m / n_bins, (m + 1) / n_bins
#         for i in range(py.shape[0]):
#             if py_value[i] > a and py_value[i] <= b:
#                 Bm[m] += 1
#                 if py_index[i] == y_test[i]:
#                     acc[m] += 1
#                 conf[m] += py_value[i]
#         if Bm[m] != 0:
#             acc[m] = acc[m] / Bm[m]
#             conf[m] = conf[m] / Bm[m]
#     ece = 0
#     for m in range(n_bins):
#         ece += Bm[m] * np.abs((acc[m] - conf[m]))
#     return ece / sum(Bm)


if __name__ == "__main__":
    test = ECE()
    extras = [{"likelihood": {"true": 0.6, "false": 0.3}},
              {"likelihood": {"true": 0.3, "false": 0.6, "neither": 0.1}},
              {"likelihood": {"true": 0.6, "false": 0.3, "neither": 0.1}}]
    references = [["true"], ["true"], ["false"]]
    print(test.compute(None, references, extras))