from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
import re
from collections import defaultdict
import json

from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score


class CFQSet:

    def __init__(self, path, image_path, transform):
        self.path = path
        self.image_path = image_path
        self.thresholds = [-1, -0.67, -0.34, -0.01, 0.33, 0.66, 0.99]

        judgment_path = os.path.join(path, "judgments.tsv")
        df = pd.read_csv(judgment, delimiter="\t")

        df["accurate"] = df["accurate"].apply(str.lower)
        df["similar"] = df["similar"].apply(str.lower)

        df["accurate"].replace("yes", 1, inplace=True)
        df["accurate"].replace("not sure", 0, inplace=True)
        df["accurate"].replace("no", -1, inplace=True)

        df["similar"].replace("reasonable", 1, inplace=True)
        df["similar"].replace("somewhat reasonable", 0, inplace=True)
        df["similar"].replace("not reasonable", -1, inplace=True)


        self.subsets = {name: CFQ(
            df[df["subset"] == name,
            image_path, transform, self.thresholds
            ) for name in df["subset"].unique()}

        self.data = df

        equiv_cap_path = os.path.join(path, "equivalent_captions.json")
        with open(equiv_cap_path, "r") as fh:
            self.equivalent_captions = json.load(fh)


    def compute_metrics(self, model, with_dots=True):
        by_subset = {key: subset.augmented_compute_metrics(
                        model, self.equivalent_captions)
                     for key, subset in self.subsets.items()}

        # aggregate over subsets
        aggs = {}
        for met in by_subset["broad"][0]:
            sums = defaultdict(float)
            normalizer = 0
            for datagroup, (values, _) in by_subset.items():
                met_vals = values[met]
                weight = len(self.subsets[datagroup].aggregated["query_hash"].drop_duplicates())
                if "accurate_-1" not in met_vals:
                    # only weighted avg for binary-decision metrics
                    weight = 1.
                    normalizer = 1.
                else:
                    normalizer += weight
                for key, val in met_vals.items():
                    sums[key] += weight * val

            aggs[met] = {key: ss / normalizer for key, ss in sums.items()}

        by_subset["weighted_average"] = aggs, None

        if not with_dots:
            by_subset = {key: val[0] for key, val in by_subset.items()}

        return by_subset

    def update_metrics(self, models, metrics=None, similarities=None):
        if metrics is None:
            metrics = defaultdict(lambda: defaultdict(dict))
        if similarities is None:
            similarities = defaultdict(dict)
        for name, mdl in models.items():
            print(name)
            results = self.compute_metrics(mdl)
            for datagroup in results:
                all_scores, dots = results[datagroup]
                similarities[name][datagroup] = dots
                for met in all_scores:
                    metrics[met][name][datagroup] = all_scores[met]

        return metrics, similarities

    def get_primary_metrics(self, metrics):
        average = metrics["weighted_average"]
        primary = {}
        primary["accurate mAP"] = average["m_ap"]["accurate_0"]
        primary["similar mAP"] = average["m_ap"]["similar_-2/3"]
        primary["nDCG"] = np.mean(list(average["linear_ndcg"].values()))
        return primary


class CFQ:

    def __init__(self, dataframe, image_dir, transform, thresholds):

        aggregated = dataframe.groupby(["query_hash", "caption", "catalog_hash"])
        aggregated = aggregated.agg({"accurate": "mean", "similar": "mean"})
        aggregated = aggregated.reset_index()
        # for sorting by accuracy first, then similarity secondarily
        # ties are (arbitrarily) broken by catalog id
        aggregated["acc_first_sorter"] = aggregated["accurate"] + 0.1 * aggregated["similar"] + aggregated["catalog_id"].iloc[0] / 10**20
        # measure relevance by sum of accuracy and similarity. for nDCG, etc
        aggregated["relevance"] = aggregated["accurate"] + aggregated["similar"] + 2
        aggregated["relevance"] = (3 * aggregated["relevance"]).astype(int)

        self.thresholds = thresholds

        for thresh in self.thresholds:
            pretty_thresh = int(np.round(thresh * 3))
            if pretty_thresh in [-3, 0, 3]:
                pretty_thresh = str(pretty_thresh // 3)
            else:
                pretty_thresh = str(pretty_thresh) + "/3"

            acc_label = f"accurate_{pretty_thresh}"
            aggregated[acc_label] = aggregated["accurate"] > thresh
            sim_label = f"similar_{pretty_thresh}"
            aggregated[sim_label] = aggregated["similar"] > thresh

        self.df = dataframe
        self.aggregated = aggregated
        self.image_dir = image_dir
        self.transform = transform

    def load_image(self, pid, preprocess=None):
        fname = os.path.join(self.image_dir, str(pid)) + ".jpeg"
        img = Image.open(fname)
        if preprocess is not None:
            return preprocess(img).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        return torch.stack([img]).float()

    def get_embeddings(self, model, queries, catalog, use_titles=False):
        if use_titles:
            raise NotImplementedError
        if hasattr(model, "preprocess"):
            preprocess = model.preprocess
        else:
            preprocess = None
        que_emb = {}
        for ii in range(len(queries)):
            que = queries.iloc[ii]
            que_id = que["query_hash"]
            que_img = self.load_image(que_id, preprocess).to(model.device)
            caption = que["caption"]
            emb = model(que_img, [caption])[0]
            que_emb[str(que_id)+caption] = emb
        cat_emb = {}
        for ii in range(len(catalog)):
            item = catalog.iloc[ii]["catalog_hash"]
            img = self.load_image(item, preprocess).to(model.device)
            emb = model(img, [None])[0]
            cat_emb[item] = emb
        return que_emb, cat_emb

    def compute_metrics(self, model, df=None):
        if df is None:
            df = self.df
        queries = df[["caption", "query_hash"]].drop_duplicates()
        catalog = df[["catalog_hash"]].drop_duplicates()
        with torch.no_grad():
            que_emb, cat_emb = self.get_embeddings(model, queries, catalog, use_titles=use_titles)

        def get_dot(row):
            key = str(row["query_hash"]) + row["caption"]
            que = que_emb[key]
            cat = cat_emb[row["catalog_hash"]]
            dot = (que @ cat) / (torch.linalg.norm(que) * torch.linalg.norm(cat))
            return dot.item()

        agg = self.aggregated.copy()
        agg["dots"] = self.aggregated.apply(get_dot, axis=1)

        metrics = defaultdict(dict)
        by_caption = agg.groupby("caption")
        # binary classification/retrieval metrics
        for col in agg.columns:
            if "accurate_" in col or "similar_" in col:
                roc_auc = AreaUnderROC(col, "dots")
                ap = AveragePrecision(col, "dots")
                rprec = RPrecision(col, "dots")

                metrics["roc_auc"][col] = roc_auc(agg)

                roc_by_caption = by_caption.apply(roc_auc)
                metrics["m_auc"][col] = roc_by_caption[roc_by_caption != -1].mean()  # conditional removes trivial

                metrics["ap"][col] = ap(agg)
                ap_by_caption = by_caption.apply(ap)
                metrics["m_ap"][col] = ap_by_caption[ap_by_caption != -1].mean()

                rprec_by_caption = by_caption.apply(rprec)
                metrics["rprec"][col] = rprec_by_caption[rprec_by_caption != -1].mean()

        # ranking metrics on unthresholded judgments
        ndcg = NormalizedDCG("relevance", "dots")
        exp_ndcg = NormalizedDCG("relevance", "dots", exponential=True)
        metrics["linear_ndcg"] = by_caption.apply(ndcg)
        metrics["exp_ndcg"] = by_caption.apply(exp_ndcg)

        return metrics, agg["dots"]

    def augmented_compute_metrics(self, model, equivalent_captions):
        df = self.df
        num_equiv = len(list(equivalent_captions.values())[0])
        query_maps = [{que: extras[ii] for que, extras in equivalent_captions.items()}
                      for ii in range(num_equiv)]
        extras = []
        for qmap in query_maps:
            extras.append(df.replace(to_replace=qmap))
        combined = pd.concat([df] + extras)

        return self.compute_metrics(model, df=combined, use_titles=use_titles)

    def captions_by_diff(self, dots, split):
        res_df = self.aggregated.copy()
        res_df["score"] = dots
        res_df["correct"] = split
        pivoted = pd.pivot_table(res_df, values="score", index="caption", columns=["correct"], aggfunc=np.mean)
        mean_score = dots.mean()
        pivoted["diff"] = (pivoted[True] - pivoted[False]) / mean_score
        return pivoted.sort_values("diff")

    def roc_by_caption(self, dots, correct):
        res_df = self.aggregated.copy()
        res_df["score"] = dots
        res_df["correct"] = correct
        def roc_agg(df):
            return roc_auc_score(df["correct"], df["score"])
        result = res_df.groupby("caption").apply(roc_agg)
        return result

    def num_at_thresholds(self):
        acc_nums = []
        sim_nums = []
        for col in self.aggregated:
            if "accurate_" in col:
                acc_nums.append(self.aggregated[col].mean())
            elif "similar_" in col:
                sim_nums.append(self.aggregated[col].mean())
        return acc_nums, sim_nums


class Metric:

    def __init__(self, ground_truth_column, score_column):
        self.ground_truth = ground_truth_column
        self.score = score_column

    def __call__(self, dataframe):
        raise NotImplementedError


class AreaUnderROC(Metric):

    def __call__(self, dataframe):
        try:
            return roc_auc_score(dataframe[self.ground_truth], dataframe[self.score])
        except ValueError as ve:
            if ve.args[0] == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                # if the problem is trivial, mark it with -1 for later removal
                return -1
            raise ve


class AveragePrecision(Metric):

    def __call__(self, dataframe):
        if len(dataframe[self.ground_truth].unique()) < 2:
            # if the set is trivial, mark it with -1 for later removal
            return -1
        return average_precision_score(dataframe[self.ground_truth], dataframe[self.score])


class RPrecision(Metric):

    def __call__(self, dataframe):
        truth = dataframe[self.ground_truth]
        scores = dataframe[self.score]
        num_true = truth.sum()
        if num_true == 0:
            return -1
        sorter = np.argsort(scores.to_numpy())[::-1]  # highest scores first
        return truth.iloc[sorter[:num_true]].sum()


class NormalizedDCG(Metric):

    def __init__(self, ground_truth_column, score_column, exponential=False):
        self.ground_truth = ground_truth_column
        self.score = score_column
        self.exponential = exponential

    def __call__(self, dataframe):
        relevance = dataframe[self.ground_truth].to_numpy()[None, :]
        if self.exponential:
            relevance = 2**relevance - 1
        dots = dataframe[self.score].to_numpy()[None, :]
        return ndcg_score(relevance, dots)
