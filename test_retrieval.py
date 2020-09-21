# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modified for experiments in https://arxiv.org/abs/2007.00145

"""Evaluates the retrieval model."""
import numpy as np
import torch
from tqdm import tqdm as tqdm
import os
import json


def test(opt, model, testset, filter_categories=False):
    if filter_categories:
        out = []
        for category in testset.categories:
            print("Evaluating on", category)
            cat_out = _test(opt, model, testset, category)
            out += [[category+name, val] for name, val in cat_out]
    else:
        out = _test(opt, model, testset)
    return out


def _test(opt, model, testset, category=None, trnsize=1000, gallery=None):
    """Tests a model over the given testset."""
    model.eval()
    if category is not None:
        test_queries = testset.get_test_queries(category=category)
    else:
        test_queries = testset.get_test_queries()

    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []
    if test_queries:
        # compute test query features
        imgs = []
        mods = []
        for t in tqdm(test_queries):
            imgs += [testset.get_img(t['source_img_id'])]
            mods += [t['mod']['str']]
            if len(imgs) >= opt.batch_size or t is test_queries[-1]:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()

                f = model.compose_img_text(imgs, mods).data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
        all_queries = np.concatenate(all_queries)
        all_target_captions = [tq['target_caption'] for tq in test_queries]

        # compute all image features (within category if applicable)
        all_imgs, all_captions = \
            compute_db_features(opt, model, testset, category)

    else:
        # use training queries to approximate training retrieval performance
        # TODO: test that this is doing what it says
        imgs0 = []
        imgs = []
        mods = []
        for i in tqdm(range(trnsize)):
            item = testset[i]
            imgs += [item["source_img_data"]]
            mods += [item['mod']['str']]
            if len(imgs) > opt.batch_size or i == (trnsize-1):
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs)

                f = model.compose_img_text(imgs.cuda(), mods).data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
            imgs0 += [item['target_img_data']]
            if len(imgs0) > opt.batch_size or i == (trnsize-1):
                imgs0 = torch.stack(imgs0).float()
                imgs0 = torch.autograd.Variable(imgs0)
                imgs0 = model.extract_img_feature(imgs0.cuda()).data.cpu().numpy()
                all_imgs += [imgs0]
                imgs0 = []
            all_captions += [item['target_caption']]
            all_target_captions += [item['target_caption']]
        all_imgs = np.concatenate(all_imgs)
        all_queries = np.concatenate(all_queries)

    nn_result, sorted_sims = nn_and_sims(testset, all_queries, all_imgs,
                                         test_queries, all_captions,
                                         category=category)

    # compute recalls
    out = []
    for k in [1, 5, 10, 50, 100]:
        recall = 0.0
        for i, nns in enumerate(nn_result):
            if all_target_captions[i] in nns[:k]:
                recall += 1
        recall /= len(nn_result)
        out += [('recall_top' + str(k) + '_correct_composition', recall)]

        if opt.dataset == 'mitstates':
            recall = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
                    recall += 1
            recall /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_adj', recall)]

            recall = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
                    recall += 1
            recall /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_noun', recall)]

    if opt.dataset == "fashioniq" and test_queries:
        dirname = os.path.join(opt.savedir, opt.exp_name, "val")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fname = os.path.join(dirname, "{}.predict.json".format(category))
        write_fashioniq(nn_result, testset, test_queries,
                        sorted_sims, fname, category)

    return out


def nn_and_sims(testset, all_queries, all_imgs, test_queries,
                all_captions, category=None):
    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    if test_queries:
        for i, t in enumerate(test_queries):
            source_id = t['source_img_id']
            if category is not None:
                # get index within category
                source_id = testset.imgs[source_id]["category"][category]
            sims[i, source_id] = -10e10  # remove query image
    nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])]
    nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]

    sorted_sims = [np.sort(sims[ii, :])[::-1] for ii in range(sims.shape[0])]

    return nn_result, sorted_sims


def predict(opt, model, testset, filter_categories=False):
    if filter_categories:
        for category in testset.categories:
            print("Evaluating on ", category)
            _predict(opt, model, testset, category)
    else:
        _predict(opt, model, testset)


def compute_db_features(opt, model, testset, category=None):
    """Compute all image features."""
    all_imgs = []
    imgs = []
    imset = testset.imgs if category is None else testset.img_by_cat[category]
    for entry in tqdm(imset):
        ind = entry["image_id"]
        imgs += [testset.get_img(ind)]
        if len(imgs) >= opt.batch_size or entry is imset[-1]:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float()
            imgs = torch.autograd.Variable(imgs).cuda()
            imgs = model.extract_img_feature(imgs).data.cpu().numpy()
            all_imgs += [imgs]
            imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [entry["captions"][0] for entry in imset]
    return all_imgs, all_captions


def compute_query_features(opt, model, testset, test_queries):
    # compute test query features
    all_queries = []
    imgs = []
    mods = []
    for t in tqdm(test_queries):
        imgs += [testset.get_img(t['source_img_id'])]
        mods += [t['mod']['str']]
        if len(imgs) >= opt.batch_size or t is test_queries[-1]:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float()
            imgs = torch.autograd.Variable(imgs).cuda()

            f = model.compose_img_text(imgs, mods).data.cpu().numpy()
            all_queries += [f]
            imgs = []
            mods = []
    all_queries = np.concatenate(all_queries)
    return all_queries



def _predict(opt, model, testset, category=None):
    model.eval()
    if category is not None:
        test_queries = testset.get_test_queries(category=category)
    else:
        test_queries = testset.get_test_queries()

    all_queries = compute_query_features(opt, model, testset, test_queries)

    all_imgs, all_captions = \
        compute_db_features(opt, model, testset, category=category)

    nn_result, sorted_sims = nn_and_sims(testset, all_queries, all_imgs,
                                         test_queries, all_captions,
                                         category=category)


    if opt.dataset == "fashioniq":
        dirname = os.path.join(opt.savedir, opt.exp_name, "test")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fname = os.path.join(dirname, "{}.predict.json".format(category))
        write_fashioniq(nn_result, testset, test_queries,
                        sorted_sims, fname, category)


def write_fashioniq(results, testset, test_queries, scores, fname, category,
                    num_to_keep=100):
    try:
        output = []
        for que, res, sc in zip(test_queries, results, scores):
            que_asin = testset.imgs[que["source_img_id"]]["asin"]
            res_asin = [testset.imgs[res[ii]]["asin"]
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
