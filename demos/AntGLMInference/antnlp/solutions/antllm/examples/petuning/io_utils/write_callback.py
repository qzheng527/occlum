from pytorch_lightning.callbacks import BasePredictionWriter
import pandas as pd
from pypai.io import TableWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_table, write_interval, extra_fields=[]):
        super().__init__(write_interval)
        self.output_table = output_table
        self.extra_fields = extra_fields
            
        self.writer = TableWriter(self.output_table, drop_if_exists=False)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        import numpy as np

        result = (
            np.asarray(predictions)
            .reshape(
                [
                    -1,
                ]
            )
            .tolist()
        )
        self.result_dict = self.init_res_dict()

        if isinstance(result, list):
            for r in result:
                self.update_res_dict(r)

            self.writer.to_table(pd.DataFrame(self.result_dict))

    def init_res_dict(self, input_field="question", output_field="answer"):
        result_dict = {output_field: [], input_field: []}
        for k in self.extra_fields:
            result_dict[k] = []
        return result_dict

    def update_res_dict(self, records, input_field="question", output_field="answer"):
        self.result_dict[output_field] += records[output_field]
        self.result_dict[input_field] += records[input_field]
        for k in self.extra_fields:
            self.result_dict[k] += list(map(lambda x: str(x), records[k]))
