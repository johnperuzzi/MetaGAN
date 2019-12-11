#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example shows how to use higher to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400

This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py
"""

import argparse
import time
import typing

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import higher

from support.omniglot_loaders import OmniglotNShot
from support.self_learned_net import SelfLearnedNet
from torch.utils.tensorboard import SummaryWriter


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument(
        '--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument(
        '--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument(
        '--task_num',
        type=int,
        help='meta batch size, namely task num',
        default=32)
    argparser.add_argument('--seed', type=int, help='random seed', default=1)
    argparser.add_argument('-v', '--verbose', action="store_true", default=False)
    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Set up the Omniglot loader.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    db = OmniglotNShot(
        '/tmp/omniglot-data',
        batchsz=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        imgsz=28,
        device=device,
    )

    # Create a vanilla PyTorch neural network that will be
    # automatically monkey-patched by higher later.
    # Before higher, models could *not* be created like this
    # and the parameters needed to be manually updated and copied
    # for the updates.
    # net = nn.Sequential(
    #     nn.Conv2d(1, 64, 3),
    #     nn.BatchNorm2d(64, momentum=1, affine=True),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(2, 2),
    #     nn.Conv2d(64, 64, 3),
    #     nn.BatchNorm2d(64, momentum=1, affine=True),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(2, 2),
    #     nn.Conv2d(64, 64, 3),
    #     nn.BatchNorm2d(64, momentum=1, affine=True),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(2, 2),
    #     Flatten(),
    #     nn.Linear(64, args.n_way)).to(device)

    net = SelfLearnedNet(args.n_way, device)

    cost_net = nn.Sequential( # removed flatten
            nn.Linear(64*3*3 + 1, 128), # plus one for the concat of 0/1
            nn.BatchNorm1d(128, momentum=1, affine=True),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(128, 64), 
            # nn.BatchNorm1d(64, momentum=1, affine=True),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32, momentum=1, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 1),
            # nn.BatchNorm1d(1, momentum=1, affine=True)
            ).to(device)
            # maybe add batch norm to end to keep around 1?

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam([{'params' : net.parameters()},
                        {'params' : cost_net.parameters()}], lr=1e-3)

    writer = {"writer" : SummaryWriter(), "train_idx" : 0, "test_idx" : 0}

    log = []
    for epoch in range(100):
        train(db, net, cost_net, device, meta_opt, epoch, log, args.verbose, writer)
        test(db, net, cost_net, device, epoch, log, args.verbose, writer)
        plot(log, args)

    writer['writer'].close()

def run_inner(x, y, n_inner_iter, fnet, diffopt, cost_net, verbose):
    learned_costs = []
    spt_losses = []
    for _ in range(n_inner_iter):
        spt_logits, shared_activations, gen_activations = fnet(x)
        spt_loss = F.cross_entropy(spt_logits, y)

        shared_activations = Flatten()(shared_activations)
        gen_activations = Flatten()(gen_activations)

        shared_activations = torch.cat([shared_activations, torch.ones(shared_activations.size()[0], 1)], dim=-1)
        gen_activations = torch.cat([gen_activations, torch.zeros(gen_activations.size()[0], 1)], dim=-1)


        learned_cost = cost_net(shared_activations)
        learned_cost = torch.mean(learned_cost)

        tot_loss = spt_loss + learned_cost

        # logging
        spt_losses.append(spt_loss.detach())
        learned_costs.append(learned_cost.detach())
        if verbose:
            print(spt_loss.detach(), learned_cost.detach())

        diffopt.step(tot_loss)
    return spt_losses, learned_costs

def train(db, net, cost_net, device, meta_opt, epoch, log, verbose, writer):
    net.train()
    n_train_iter = db.x_train.shape[0] // db.batchsz

    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next()

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        for i in range(task_num):
            if verbose:
                print("task: " + str(i))
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                spt_losses, learned_costs = run_inner(x_spt[i], y_spt[i], n_inner_iter, fnet, diffopt, cost_net, verbose)

                for spt_loss, learned_cost in zip(spt_losses, learned_costs):
                    writer['writer'].add_scalar('Train/spt_losses', spt_loss, writer['train_idx'])
                    writer['writer'].add_scalar('Train/learned_costs', learned_cost, writer['train_idx'])
                    writer['train_idx'] += 1 

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = fnet(x_qry[i], cost=False)
                qry_loss = F.cross_entropy(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(
                    dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward()

        meta_opt.step()
        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100. * sum(qry_accs) / task_num
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(
                f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
            )

        log.append({
            'epoch': i,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
            'time': time.time(),
        })


def test(db, net, cost_net, device, epoch, log, verbose, writer):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    net.train()
    n_test_iter = db.x_test.shape[0] // db.batchsz

    qry_losses = []
    qry_accs = []

    for batch_idx in range(n_test_iter):
        x_spt, y_spt, x_qry, y_qry = db.next('test')


        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                spt_losses, learned_costs = run_inner(x_spt[i], y_spt[i], n_inner_iter, fnet, diffopt, cost_net, verbose)
                for spt_loss, learned_cost in zip(spt_losses, learned_costs):
                    writer['writer'].add_scalar('Test/spt_losses', spt_loss, writer['test_idx'])
                    writer['writer'].add_scalar('Test/learned_costs', learned_cost, writer['test_idx'])
                    writer['test_idx'] += 1 

                # The query loss and acc induced by these parameters.
                qry_logits = fnet(x_qry[i], cost=False).detach()
                qry_loss = F.cross_entropy(
                    qry_logits, y_qry[i], reduction='none')
                qry_losses.append(qry_loss.detach())
                qry_accs.append(
                    (qry_logits.argmax(dim=1) == y_qry[i]).detach())

    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
    print(
        f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}'
    )
    log.append({
        'epoch': epoch + 1,
        'loss': qry_losses,
        'acc': qry_accs,
        'mode': 'test',
        'time': time.time(),
    })




def plot(log, args):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'maml-accs_nway_' + str(args.n_way) + "_k_shot_" + str(args.k_spt) + "_k_qry_" + str(args.k_qry)+ '.png' 
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


# Won't need this after this PR is merged in:
# https://github.com/pytorch/pytorch/pull/22245
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


if __name__ == '__main__':
    main()
