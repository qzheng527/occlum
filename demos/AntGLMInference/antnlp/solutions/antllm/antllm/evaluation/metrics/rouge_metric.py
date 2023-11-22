# coding=utf-8
# @Author: siye.lsy
# Copyright 2021 Ant Group Co., Ltd.
# from overrides import overrides
from rouge import Rouge


class ROUGE(Rouge):
    '''
    修改自 rouge.Rouge, 移除了原有代码用`.`进行句子分割的逻辑
    '''
    def compute_score(self, refs, hyps):
        '''
        计算ROUGE得分

        Parameters
        ----------
        refs: list(str)
            参考文本, 用空格进行分割. 例如 ["你 好", "我 很 OK"]
        hyps: list(str)
            生成文本, 用空格进行分割. 例如 ["你 好", "我 很 OK"]

        Returns
        -------
        scores: dict
            {"ROUGE-1": {"R": 1.0, "P": 1.0, "F": 1.0}}
        '''
        assert len(refs) == len(hyps)
        raw_scores = self.get_scores(hyps, refs, avg=True)

        scores = {}
        for k in raw_scores:
            v = raw_scores[k]
            k = k.upper()
            scores[k] = {
                "R": v["r"],
                "P": v["p"],
                "F": v["f"]
            }
        return scores

    # @overrides
    def _get_scores(self, hyps, refs):
        scores = []
        for hyp, ref in zip(hyps, refs):
            sen_score = {}

            hyp = [hyp]
            ref = [ref]

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(
                    hyp,
                    ref,
                    raw_results=self.raw_results,
                    exclusive=self.exclusive)
                sen_score[m] = {s: sc[s] for s in self.stats}

            if self.return_lengths:
                lengths = {
                    "hyp": len(" ".join(hyp).split()),
                    "ref": len(" ".join(ref).split())
                }
                sen_score["lengths"] = lengths
            scores.append(sen_score)
        return scores

    # @overrides
    def _get_avg_scores(self, hyps, refs):
        scores = {m: {s: 0 for s in self.stats} for m in self.metrics}
        if self.return_lengths:
            scores["lengths"] = {"hyp": 0, "ref": 0}

        count = 0
        for (hyp, ref) in zip(hyps, refs):
            hyp = [hyp]
            ref = [ref]

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref, exclusive=self.exclusive)
                scores[m] = {s: scores[m][s] + sc[s] for s in self.stats}

            if self.return_lengths:
                scores["lengths"]["hyp"] += len(" ".join(hyp).split())
                scores["lengths"]["ref"] += len(" ".join(ref).split())

            count += 1
        avg_scores = {
            m: {s: scores[m][s] / count for s in self.stats}
            for m in self.metrics
        }

        if self.return_lengths:
            avg_scores["lengths"] = {
                k: scores["lengths"][k] / count
                for k in ["hyp", "ref"]
            }

        return avg_scores


if __name__ == "__main__":
    rouge = ROUGE()
    hyps, refs = [], []
    import json
    with open('results.json') as f:
        for line in f:
            h = json.loads(line)
            refs.append(h['ref'])
            hyps.append(h['pred'])
    print(rouge.compute_score(refs, hyps))
