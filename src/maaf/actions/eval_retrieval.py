# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


"""Evaluates the retrieval model."""
import numpy as np
import torch
from ..utils.misc_utils import tqdm  # with dynamic_ncols=True
import os
import json


def test(cfg, model, testset, filter_categories=False):
    if filter_categories:
        out = []
        for category in testset.categories:
            print("Evaluating on", category)
            cat_out = _test(cfg, model, testset, category)
            out += [[category + name, val] for name, val in cat_out]
    else:
        out = _test(cfg, model, testset)

    if cfg.DATASET.NAME == "fashioniq":
        scores = [metric for name, metric in out if
                  "top100" not in name and ("top10" in name or "top50" in name)]
        out += [["fiq_score", np.mean(scores)]]
    return out


def _test(cfg, model, testset, category=None):
    """Tests a model over the given testset."""
    model.eval()

    all_queries = compute_query_features(cfg, model, testset,
                                         category=category)
    if hasattr(testset, "test_queries"):
        test_queries = testset.test_queries
        all_targets = [que["target_caption"] for que in testset.test_queries]
    else:
        test_queries = testset.data
        if category is None:
            all_targets = [tq['target_id'] for tq in test_queries]
        else:
            all_targets = [testset.gallery[tq['target_id']]["category"][category]
                           for tq in testset.data_by_category[category]]

    # compute all gallery features (within category if applicable)
    gallery, all_labels = \
        compute_db_features(cfg, model, testset, category=category)
    if hasattr(testset, "test_queries"):
        all_labels = [dd["captions"][0] for dd in testset.data]

    nn_result, sorted_sims = nn_and_sims(testset, all_queries, gallery,
                                         all_labels, category=category)

    # compute recalls
    out = []
    for k in [1, 5, 10, 50, 100]:
        recall = 0.0
        for i, nns in enumerate(nn_result):
            if all_targets[i] in nns[:k]:
                recall += 1
        recall /= len(nn_result)
        out += [('_test_recall_top' + str(k), recall)]

    if cfg.DATASET.NAME == "fashioniq":
        dirname = os.path.join(cfg.OUTPUT_DIR, cfg.EXP_NAME, "val")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fname = os.path.join(dirname, "{}.predict.json".format(category))
        write_fashioniq(nn_result, testset, sorted_sims, fname, category)

    return out


def nn_and_sims(testset, all_queries, gallery,
                all_labels, category=None):
    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(gallery.T)
    data = testset.data if category is None else testset.data_by_category[category]
    for ii, entry in enumerate(data):
        if "source_id" in entry:
            source_id = entry["source_id"]
            if category is not None:
                # get index within category
                source_id = testset.gallery.gallery[source_id]["category"][category]
            sims[ii, source_id] = -10e10  # remove query image

    nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])]
    nn_result = [[all_labels[nn] for nn in nns] for nns in nn_result]

    sorted_sims = [np.sort(sims[ii, :])[::-1] for ii in range(sims.shape[0])]

    return nn_result, sorted_sims

def compute_db_features(cfg, model, testset, category=None):
    """Compute all gallery features."""
    all_feat = []
    if category is None:
        loader = testset.get_gallery_loader(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            num_workers=cfg.DATA_LOADER.LOADER_NUM_WORKERS)
    else:
        loader = testset.get_gallery_loader(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            num_workers=cfg.DATA_LOADER.LOADER_NUM_WORKERS,
            category=category)

    for batch in tqdm(loader):
        images = [dd["target_image"] for dd in batch]
        if len(images) > 0:
            if hasattr(model, "image_transform"):
                images = [model.image_transform(im) for im in images]
            images = torch.stack(images).float().to(model.device)
        texts = [dd["target_text"] for dd in batch]

        emb = model(images, texts).data.cpu().numpy()
        all_feat += [emb]

    all_feat = np.concatenate(all_feat)
    all_labels = list(range(len(testset.gallery)))
    return all_feat, all_labels


def compute_query_features(cfg, model, testset, category=None):
    if category is None:
        loader = testset.get_loader(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.DATA_LOADER.LOADER_NUM_WORKERS)
    else:
        loader = testset.get_loader(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.DATA_LOADER.LOADER_NUM_WORKERS,
            category=category)

    all_queries = []

    # compute query/source features
    for batch in tqdm(loader):
        source_img = [dd['source_image'] for dd in batch]
        if len(source_img) > 0:
            if hasattr(model, "image_transform"):
                source_img = [model.image_transform(im) for im in source_img]
            source_img = torch.stack(source_img).float().to(model.device)
        source_text = [dd["source_text"] for dd in batch]

        query_emb = model(source_img, source_text).data.cpu().numpy()
        all_queries += [query_emb]

    return np.concatenate(all_queries)


def predict(cfg, model, testset, filter_categories=False):
    if filter_categories:
        for category in testset.categories:
            print("Evaluating on ", category)
            _predict(cfg, model, testset, category)
    else:
        _predict(cfg, model, testset)


def _predict(cfg, model, testset, category=None):
    model.eval()

    all_queries = compute_query_features(cfg, model, testset)

    gallery, all_labels = \
        compute_db_features(cfg, model, testset, category=category)

    nn_result, sorted_sims = nn_and_sims(testset, all_queries, gallery,
                                         all_labels, category=category)


    if cfg.DATASET.NAME == "fashioniq":
        dirname = os.path.join(cfg.OUTPUT_DIR, cfg.EXP_NAME, "test")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fname = os.path.join(dirname, "{}.predict.json".format(category))
        write_fashioniq(nn_result, testset, sorted_sims, fname, category)


def write_fashioniq(results, testset, scores, fname, category,
                    num_to_keep=100):
    try:
        output = []
        for que, res, sc in zip(testset.data_by_category[category], results, scores):
            que_asin = testset.gallery.gallery[que["source_id"]]["asin"]
            res_asin = [testset.gallery.gallery_by_cat[category].gallery[res[ii]]["asin"]
                        for ii in range(num_to_keep)]
            entry = {"candidate": str(que_asin),
                     "ranking": [str(ra) for ra in res_asin],
                     "scores": sc.tolist()[:num_to_keep]}
            output.append(entry)

        with open(fname, "w") as fh:
            json.dump(output, fh)
    except BaseException:
        print("Error in write_fashioniq")
        import IPython; IPython.embed()
    else:
        print("wrote to", fname)


def ndcg(relevances, all_bad=1):
    """When all judgments are 0, return all_bad. Otherwise, return NDCG."""
    dcg = relevances / np.log2(np.arange(len(relevances)) + 2)
    dcg = np.sum(dcg)
    sorted_rel = np.sort(relevances)[::-1]
    max_dcg = sorted_rel / np.log2(np.arange(len(sorted_rel)) + 2)
    max_dcg = np.sum(max_dcg)
    if max_dcg == 0:
        return all_bad
    else:
        return dcg / max_dcg


def test_ndcg(cfg, model, dataset_dict):
    testset = dataset_dict["test"]
    all_query_emb = compute_query_features(cfg, model, testset)
    gallery, all_labels = compute_db_features(cfg, model, testset)

    ndcg_values = []
    for query_data in [testset.head_query_data, testset.random_query_data]:
        aggregate_ndcg = []
        for query, entries in query_data.items():
            gallery_indices = [ent["target_id"] for ent in entries]
            gallery_embs = [gallery[idx] for idx in gallery_indices]
            query_emb = all_query_emb[testset.query_to_index[query]]
            dots = np.dot(gallery_embs, query_emb)
            sorter = np.argsort(dots)[::-1]
            judgments = [entries[ii]["judgment"] for ii in sorter]
            relevances = [0 if jj == "Bad" else 1 for jj in judgments]
            this_ndcg = ndcg(np.array(relevances))
            aggregate_ndcg.append(this_ndcg)

        ndcg_values.append(np.mean(aggregate_ndcg))

    out = [("head_queries_ndcg", ndcg_values[0]),
           ("random_queries_ndcg", ndcg_values[1])]

    return out

def test_paired(testset, cfg, model):
    """
    Retrieval evaluation suitable when data is entirely source-target pairs.
    Compute recall@k for several k, and relative rank of pairs.
    """
    model.eval()

    # compute query/source and target features
    all_queries = []
    all_targets = []
    loader = testset.get_loader(cfg.SOLVER.BATCH_SIZE)
    for batch in tqdm(loader):
        source_img = [dd['source_image'] for dd in batch]
        if source_img[0] is not None:
            if hasattr(model, "image_transform"):
                source_img = [model.image_transform(im) for im in source_img]
            source_img = torch.stack(source_img).to(model.device).float()
        source_text = [dd["source_text"] for dd in batch]

        query_emb = model(source_img, source_text).data.cpu().numpy()
        all_queries += [query_emb]

        target_img = [dd['target_image'] for dd in batch]
        if target_img[0] is not None:
            if hasattr(model, "image_transform"):
                target_img = [model.image_transform(im) for im in target_img]
            target_img = torch.stack(target_img).to(model.device).float()
        target_text = [dd["target_text"] for dd in batch]

        target_emb = model(target_img, target_text).data.cpu().numpy()
        all_targets += [target_emb]

    all_queries = np.concatenate(all_queries)
    all_targets = np.concatenate(all_targets)

    # compute similarities and nearest neighbors
    sims = all_queries.dot(all_targets.T)
    nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])]

    # compute recall@k
    out = []
    for k in [1, 5, 10, 50, 100]:
        recall = 0.0
        for i, nns in enumerate(nn_result):
            if i in nns[:k]:
                recall += 1
        recall /= len(nn_result)
        out += [('_recall_top' + str(k), recall)]

    # compute ranks of matches
    ranks = [np.where(nn_result[ii] == ii)[0][0] for ii in range(len(nn_result))]
    relative = np.array(ranks) / len(nn_result)
    out += [('_mean_rel_rank', relative.mean())]
    out += [('_median_rel_rank', np.median(relative))]

    return out
