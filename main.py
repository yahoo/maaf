# Copyright 2019 Google Inc. All Rights Reserved.
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


"""Main method to train the model.

Usage example:
python main.py --dataset=fashioniq --dataset_path=/home/sherdade/data/FashionIQ \
  --num_iters=160000 --model=tirg --loss=batch_based_classification \
  --learning_rate_decay_frequency=50000 --comment=fashioniq_tirg --freeze_img_model \
  --exp_name = debug

python main.py --dataset=fashioniq --dataset_path=/home/sherdade/data/FashionIQ\
--num_iters=120000 --model=tirg --loss=batch_based_classification\
--learning_rate_decay=.25 --learning_rate_decay_frequency=30000\
--freeze_img_model --drop_worst_flag --exp_name=debug"""


import argparse
import os
import sys
import time
from datasets.datasets import load_dataset
from models import img_text_composition_models
import numpy as np
from tensorboardX import SummaryWriter
import test_retrieval
import torch
import torch.utils.data
from tqdm import tqdm as tqdm
import git  # pip install gitpython

torch.set_num_threads(3)

def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--comment', type=str, default='test_notebook')
    parser.add_argument('--savedir', type=str, default='')
    parser.add_argument('--inspect', action="store_true")

    parser.add_argument('--dataset', type=str, default='css3d')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--model', type=str, default='tirg')
    parser.add_argument('--embed_dim', type=int, default=512)

    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--learning_rate_decay', type=float, default=0.1)
    parser.add_argument('--learning_rate_decay_frequency', type=int, default=9999999)
    parser.add_argument('--lr_decay_only_once', action="store_true")
    # more flexible learning rate scheduling. both args must be set or we default to old scheme
    parser.add_argument('--scheduled_lr_rates', type=str, default="",
        help="Separate rates by commas." +
        "The learning_rate argument sets the initial rate; " +
        "this param sets rates after each scheduled_lr_iters entry" +
        "If empty string, old regular decay schedule is used.")
    parser.add_argument('--scheduled_lr_iters', type=str, default="",
        help="Separate iteration numbers by commas." +
             "If empty string, old regular decay schedule is used.")

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--num_iters', type=int, default=210000)
    parser.add_argument('--loss', type=str, default='soft_triplet')
    parser.add_argument('--loader_num_workers', type=int, default=4)
    parser.add_argument('-t', "--test_only", action="store_true")
    parser.add_argument('-l', '--load', type=str, default="")

    parser.add_argument('--dropout_rate', type=float, default=0.1)

    parser.add_argument(
        '--drop_worst_flag', action='store_true',
        help='If added the model will ingore the highest --drop_worst_rate losses')
    parser.add_argument('--drop_worst_rate', type=float, default=0.2)

    parser.add_argument(
        '--freeze_img_model', action='store_true',
        help='If added the loaded image model weights will not be finetuned')
    parser.add_argument('--pretrained_weight_lr_factor_image', type=float, default=0.1)
    parser.add_argument('--pretrained_weight_lr_factor_text', type=float, default=1.)
    parser.add_argument('--image_model_arch', type=str, default='resnet50')
    parser.add_argument('--image_model_path', type=str, default='')
    parser.add_argument(
        '--not_pretrained', action='store_true',
        help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')

    parser.add_argument(
        '--image_with_unk_string', action='store_true',
        help='If added, images wihout modifying captions (i.e., target images) are fed through text+image composer with copies of <UNK>')

    parser.add_argument(
        '--freeze_text_model', action='store_true',
        help='If added the loaded text model weights will not be finetuned')
    parser.add_argument('--text_model_arch', type=str, default='lstm')

    parser.add_argument('--whole_model_path', type=str, default='')
    parser.add_argument('--pretraining_dataset', type=str, default='')
    parser.add_argument('--pretraining_dataset_path', type=str, default='')

    parser.add_argument('--text_model_layers', type=int, default=1)
    parser.add_argument('--threshold_rare_words', type=int, default=0)

    parser.add_argument('--number_attention_blocks', type=int, default=1)
    parser.add_argument('--width_per_attention_block', type=int, default=256)
    parser.add_argument('--number_attention_heads', type=int, default=8)
    parser.add_argument('--att_layer_spec', type=str, default="3_4")
    parser.add_argument('--attn_positional_encoding', default=None)
    parser.add_argument('--resolutionwise_pool', action='store_true')

    # specific to sequence concat attention composition
    parser.add_argument('--sequence_concat_include_text', action="store_true",
        help="use post-attn text embeddings in pooling to get final composed embedding")
    parser.add_argument('--sequence_concat_img_through_attn', action="store_true",
        help="target image pathway goes through embedding layers")
    parser.add_argument('--attn_softmax_replacement', type=str, default="none")
    parser.add_argument('--attn_2stream_mode', type=str, default="xxx_xmm_xff")

    parser.add_argument('--train_on_validation_set', action="store_true")

    parser.add_argument(
        '--save_every', type=int, default=100,
        help="keep checkpoints this often in epochs")
    parser.add_argument(
        '--eval_every', type=int, default=3,
        help="run eval on val set this often in epochs")
    parser.add_argument('--final_eval_on_test', action="store_true")

    args = parser.parse_args()
    if args.load == "":
        args.load = None
    if args.image_model_path in ["", "none", "None"]:
        args.image_model_path = None
    if args.image_model_arch in ["", "none", "None"]:
        args.image_model_arch = None
    if args.whole_model_path == '':
        args.whole_model_path = None
    return args


def create_model_and_optimizer(opt, texts):
    """Builds the model and related optimizer."""
    print('Creating model and optimizer for', opt.model)

    if opt.model == 'imgonly':
        model = img_text_composition_models.SimpleModelImageOnly(opt, texts)
    elif opt.model == 'textonly':
        model = img_text_composition_models.SimpleModelTextOnly(opt, texts)
    elif opt.model == 'attention':
        print("setting up attention model")
        model = img_text_composition_models.AttentionComposition(opt, texts)
    elif opt.model == "sequence_concat_attention":
        print("Setting up sequence concatenation attention model")
        model = img_text_composition_models.SequenceConcatAttention(opt, texts)
    elif opt.model == "seqcat_outtoken":
        model = img_text_composition_models.SeqCatWithOutputToken(opt, texts)
    elif opt.model == 'concat':
        model = img_text_composition_models.Concat(opt, texts)
    elif opt.model == 'tirg':
        model = img_text_composition_models.TIRG(opt, texts)
    elif opt.model == 'tirg_lastconv':
        model = img_text_composition_models.TIRGLastConv(opt, texts)
    elif opt.model == 'add':
        model = img_text_composition_models.Addition(opt, texts)
    else:
        print('Invalid model', opt.model)
        sys.exit()

    if opt.whole_model_path is not None:
        loaded_dict = torch.load(opt.whole_model_path)
        pre_opt = argparse.Namespace
        pre_opt.dataset = opt.pretraining_dataset
        pre_opt.dataset_path = opt.pretraining_dataset_path
        pre_opt.train_on_validation_set = False

        pretrainset = load_dataset(pre_opt)["train"]

        model.text_model.prepend_vocab(pretrainset.get_all_texts())

        model.load_state_dict(loaded_dict["model_state_dict"])
        model.text_model.consolidate_vocab()

    if opt.threshold_rare_words > 0:
        model.text_model.vocab.threshold_rare_words(opt.threshold_rare_words)
        print(model.text_model.vocab.get_size(), ' total words seen')
        print(len(set(model.text_model.vocab.word2id.values())) - 1,
              ' words seen enough times to keep')

    model = model.cuda()

    # create optimizer
    params = []
    # low learning rate for pretrained layers on real image datasets
    if opt.dataset != 'css3d':
        params.append({
            'params': [p for p in model.img_model.fc.parameters()],
            'lr': opt.learning_rate
        })
        params.append({
            'params': [p for p in model.img_model.parameters()],
            'lr': opt.pretrained_weight_lr_factor_image * opt.learning_rate
        })
        params.append({
            'params': [p for p in model.text_model.parameters()],
            'lr': opt.pretrained_weight_lr_factor_text * opt.learning_rate
        })
    params.append({'params': [p for p in model.parameters()]})
    # remove duplicated params. whichever lr was assigned first is preserved
    for _, p1 in enumerate(params):
        for _, p2 in enumerate(params):
            if p1 is not p2:
                for p11 in p1['params']:
                    for j, p22 in enumerate(p2['params']):
                        if p11 is p22:
                            p2['params'][j] = torch.tensor(
                                0.0, requires_grad=True)
    optimizer = torch.optim.SGD(
        params, lr=opt.learning_rate, momentum=0.9,
        weight_decay=opt.weight_decay)
    return model, optimizer


def run_eval(opt, logger, dataset_dict, model, it, eval_on_test=False):
    trainset = dataset_dict["train"]
    if eval_on_test:
        testset = dataset_dict["test"]
    else:
        testset = dataset_dict.get("val", dataset_dict["test"])
    tests = []
    for name, dataset in [('train', trainset), ('test', testset)]:
        categ = opt.dataset == "fashioniq" and name == 'test'
        t = test_retrieval.test(opt, model, dataset, filter_categories=categ)
        tests += [(name + ' ' + metric_name, metric_value)
                  for metric_name, metric_value in t]
    for metric_name, metric_value in tests:
        logger.add_scalar(metric_name, metric_value, it)
        print('    ', metric_name, round(metric_value, 4))
    if opt.dataset == "fashioniq":
        scores = [metric for name, metric in tests if
                  "test" in name and ("top10_" in name or "top50_" in name)]
        fiq_score = np.mean(scores)
        logger.add_scalar("fiq_score", fiq_score, it)
        print('    ', 'fiq_score', round(fiq_score, 4))


def train_loop(opt, logger, dataset_dict, model, optimizer, initial_it):
    """Function for train loop"""
    print('Begin training')
    scheduled_lr = False
    if opt.scheduled_lr_rates != "":
        scheduled_rates = [float(rate) for rate in opt.scheduled_lr_rates.split(",")]
        if opt.scheduled_lr_iters != "":
            schedule_iters = [int(itr) for itr in opt.scheduled_lr_iters.split(",")]
            scheduled_lr = True
            current_lr = opt.learning_rate

    losses_tracking = {}
    it = initial_it
    epoch = -1
    tic = time.time()
    trainset = dataset_dict["train"]
    while it < opt.num_iters:
        epoch += 1
        trainloader = trainset.get_loader(
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.loader_num_workers)

        # show/log stats
        print('It', it, 'epoch', epoch, 'Elapsed time',
            round(time.time() - tic, 4), opt.comment)
        tic = time.time()
        for loss_name in losses_tracking:
            avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
            print('    Loss', loss_name, round(avg_loss, 4))
            logger.add_scalar(loss_name, avg_loss, it)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)

        # test
        if epoch % opt.eval_every == 1 or opt.eval_every == 1:
            run_eval(opt, logger, dataset_dict, model, it)

        # save checkpoint
        torch.save({
            'it': it,
            'opt': opt,
            'model_state_dict': model.state_dict(),
        }, logger.file_writer.get_logdir() + '/latest_checkpoint.pth')

        if epoch % opt.save_every == 0 and epoch > 0:
            torch.save({
                'it': it,
                'opt': opt,
                'model_state_dict': model.state_dict()},
                logger.file_writer.get_logdir() + '/ckpt_epoch{}.pth'.format(epoch))

        # run trainning for 1 epoch
        model.train()

        def training_1_iter(data):
            assert type(data) is list
            img1 = np.stack([d['source_img_data'] for d in data])
            img1 = torch.from_numpy(img1).float().cuda()
            img2 = np.stack([d['target_img_data'] for d in data])
            img2 = torch.from_numpy(img2).float().cuda()
            mods = [str(d['mod']['str']) for d in data]

            # compute loss
            losses = []
            if opt.loss == 'soft_triplet':
                loss_value = model.compute_loss(
                  img1, mods, img2, soft_triplet_loss=True)
            elif opt.loss == 'batch_based_classification':
                loss_value = model.compute_loss(
                  img1, mods, img2, soft_triplet_loss=False)
            else:
                print('Invalid loss function', opt.loss)
                sys.exit()
            loss_name = opt.loss
            loss_weight = 1.0
            losses += [(loss_name, loss_weight, loss_value)]
            total_loss = sum([
                l_weight * l_value
                for _, l_weight, l_value in losses
            ])
            assert not torch.isnan(total_loss)
            losses += [('total training loss', None, total_loss)]

            # track losses
            for l_name, l_weight, l_value in losses:
                if l_name not in losses_tracking:
                    losses_tracking[loss_name] = []
                losses_tracking[loss_name].append(float(loss_value))

            # gradient descend
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
            it += 1
            training_1_iter(data)

            if scheduled_lr:
                if len(schedule_iters) > 0:
                    if it > schedule_iters[0]:
                        lr_factor = scheduled_rates[0] / current_lr
                        for g in optimizer.param_groups:
                            g['lr'] *= lr_factor
                        current_lr = scheduled_rates[0]

                        del schedule_iters[0]
                        del scheduled_rates[0]


            else:
                # decay learing rate by old method
                decay = False
                if it >= opt.learning_rate_decay_frequency:
                    if it == opt.learning_rate_decay_frequency:
                        decay = True
                    elif it % opt.learning_rate_decay_frequency == 0:
                        decay = not opt.lr_decay_only_once
                if decay:
                    for g in optimizer.param_groups:
                        g['lr'] *= opt.learning_rate_decay

    print('Finished training, running final eval')
    return it


def main():
    opt = parse_opt()
    print('Arguments:')
    for k in opt.__dict__.keys():
        print('    ', k, ':', str(opt.__dict__[k]))

    if opt.load is not None:
        logger = SummaryWriter(opt.load)
    else:
        #logger = SummaryWriter(comment=opt.comment)
        logger = SummaryWriter(logdir = os.path.join(opt.savedir, opt.exp_name))
        print('Log files saved to', logger.file_writer.get_logdir())
    for k in opt.__dict__.keys():
        logger.add_text(k, str(opt.__dict__[k]))

    # get and save the version of the code being run
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.add_text("git_sha", sha)

    dataset_dict = load_dataset(opt)
    model, optimizer = create_model_and_optimizer(
      opt, dataset_dict["train"].get_all_texts())

    if opt.load is not None:
        print("loading from: %s" % opt.load)
        loaded_dict = \
          torch.load(logger.file_writer.get_logdir()+"/latest_checkpoint.pth")
        model.load_state_dict(loaded_dict["model_state_dict"])
        initial_it = loaded_dict["it"]

        for g in optimizer.param_groups:
            print('learning rate(s):')
            print(g["lr"])
            # g['lr'] *= opt.learning_rate_decay
    else:
        initial_it = 0

    if opt.inspect:
        import IPython
        IPython.embed()

    if not opt.test_only:
        final_it = train_loop(opt, logger, dataset_dict, model,
                              optimizer, initial_it)
    else:
        final_it = initial_it
    run_eval(opt, logger, dataset_dict, model,
             final_it + int(opt.final_eval_on_test),
             eval_on_test=opt.final_eval_on_test)
    if opt.dataset == "fashioniq":
        print('Generating FashionIQ submission...')
        test_retrieval.predict(opt, model, dataset_dict["test"],
                               filter_categories=True)
        print('done')

    logger.close()


if __name__ == '__main__':
    main()
