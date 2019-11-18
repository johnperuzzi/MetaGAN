import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy
from torch.autograd import Variable



class MetaGAN(nn.Module):
    """
    Meta Learner with GAN incorporated
    """
    def __init__(self, args, shared_config, nway_config, discriminator_config):
        """

        :param args:
        """
        super(MetaGAN, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.tasks_per_batch = args.tasks_per_batch
        self.update_steps = args.update_steps
        self.update_steps_test = args.update_steps_test

        # self.generator = Generator(gen_config, args.img_c, args.img_sz, args)

        self.shared_net = Learner(shared_config, args.img_c, args.img_sz)
        self.nway_net = Learner(nway_config, args.img_c, args.img_sz)
        self.discrim_net = Learner(discriminator_config, args.img_c, args.img_sz)

        params = list(self.shared_net.parameters()) + list(self.nway_net.parameters()) + list(self.discrim_net.parameters())
        self.meta_optim = optim.Adam(params, lr=self.meta_lr)
        # self.meta_shared_optim = optim.Adam(self.shared_net.parameters(), lr=self.meta_lr)
        # self.meta_nway_optim = optim.Adam(self.nway_net.parameters(), lr=self.meta_lr)
        # self.meta_discrim_optim = optim.Adam(self.discrim_net.parameters(), lr=self.meta_lr)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    # Returns predicted class logits and descriminator outputs according
    # to the input "x", using the shared/nway/discrim nets and weights provided
    def pred(self, x, weights=[None, None, None], nets=None, discrim=True):
        if type(nets) == type(None):
            nets = [self.shared_net, self.nway_net, self.discrim_net]

        shared_weights, nway_weights, discrim_weights = weights
        shared_net, nway_net, discrim_net = nets

        shared_layer = shared_net(x, vars=shared_weights, bn_training=True)
        class_logits = nway_net(shared_layer, vars=nway_weights, bn_training=True)
        if not discrim:
            return class_logits

        discrim_preds = discrim_net(shared_layer, vars=discrim_weights, bn_training=True)
        return class_logits, discrim_preds

    # Returns the loss(es) of the y's according to the class and possibly also descriminator predictions
    def loss(self, class_logits, y_class, discrim_preds=None, y_discrim=None):
        nway_loss = F.cross_entropy(class_logits, y_class)

        if type(discrim_preds) == type(None):
            return nway_loss

        discrim_loss = F.mse_loss(discrim_preds, y_discrim)
        return nway_loss, discrim_loss

    # Returns new weights by backpropping their affect on the losses.
    # Losses and weights should be (shared, nway, descrim)
    def update_weights(self, losses, weights):
        shared_loss, nway_loss, discrim_loss = losses
        shared_weights, nway_weights, discrim_weights = weights

        n_grad = torch.autograd.grad(nway_loss, nway_weights, retain_graph=True)
        n_weights = [w - self.update_lr * grad for grad, w in zip(n_grad, nway_weights)]

        d_grad = torch.autograd.grad(discrim_loss, discrim_weights, retain_graph=True)
        d_weights = [w - self.update_lr * grad for grad, w in zip(d_grad, discrim_weights)]

        s_grad = torch.autograd.grad(shared_loss, shared_weights)
        s_weights = [w - self.update_lr * grad for grad, w in zip(s_grad, shared_weights)]

        return s_weights, n_weights, d_weights

    def single_task_forward(self, x_spt, y_spt, x_qry, y_qry, nets=None):
        support_sz, c_, h, w = x_spt.size()

        corrects = np.zeros(self.update_steps + 1)
        if type(nets) == type(None):
            nets = (self.shared_net, self.nway_net, self.discrim_net)

        # net_weights = [net.parameters() for net in nets]

        # check if I need to copy these like this or can do as above
        net_weights = []
        for net in nets:
            net_weights.append([w.clone() for w in net.parameters()])

        cuda = False
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor 

        # this is the meta-test loss and accuracy before first update
        with torch.no_grad():
            q_class_logits = self.pred(x_qry, weights=[None, None, None], discrim=False)

            pred_q = F.softmax(q_class_logits, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] += correct


        # 0. fake gen examples. uses same examples for all inner update steps
        x_gen = torch.rand_like(x_spt)
        y_gen = y_spt

        valid = Variable(FloatTensor(support_sz, 1).fill_(1.0), requires_grad=True)
        fake = Variable(FloatTensor(support_sz, 1).fill_(0.0), requires_grad=True)

        # run the i-th task and compute loss for k-th inner update
        for k in range(1, self.update_steps + 1):

            # run discriminator on real data
            real_class_logits, real_discrim_preds = self.pred(x_spt, weights=net_weights)
            real_nway_loss, real_discrim_loss = self.loss(real_class_logits, y_spt, real_discrim_preds, valid)

            # run discriminator on generated data
            gen_class_logits, gen_discrim_preds = self.pred(x_gen, weights=net_weights)
            gen_nway_loss, gen_discrim_loss = self.loss(gen_class_logits, y_gen, gen_discrim_preds, fake)

            nway_loss = (gen_nway_loss + real_nway_loss) / 2
            discrim_loss = (gen_discrim_loss + real_discrim_loss) / 2
            shared_loss = nway_loss + discrim_loss

            # 2. compute grad on theta_pi
            losses = (shared_loss, nway_loss, discrim_loss)
            net_weights = self.update_weights(losses, net_weights)

            # meta-test accuracy
            with torch.no_grad():
                # [query_sz]
                q_class_logits = self.pred(x_qry, weights=net_weights, discrim=False)

                pred_q = F.softmax(q_class_logits, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[k] += correct

        # meta-test loss
        q_class_logits = self.pred(x_qry, weights=net_weights, discrim=False)
        loss_q = self.loss(q_class_logits, y_qry) # doesn't use discrim loss

        return loss_q, corrects  


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, support_sz, c_, h, w]
        :param y_spt:   [b, support_sz]
        :param x_qry:   [b, query_sz, c_, h, w]
        :param y_qry:   [b, query_sz]
        :return:
        """
        tasks_per_batch, support_sz, c_, h, w = x_spt.size()
        query_sz = x_qry.size(1)

        loss_q = 0
        corrects = np.zeros(self.update_steps + 1)
        
        for i in range(tasks_per_batch):
            loss_q_tmp, corrects_tmp = self.single_task_forward(x_spt[i], y_spt[i], x_qry[i], y_qry[i])
            loss_q += loss_q_tmp
            corrects += corrects_tmp

        # end of all tasks
        # sum over final losses on query set across all tasks
        loss_q /= tasks_per_batch

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        accs = corrects / (query_sz * tasks_per_batch)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [support_sz, c_, h, w]
        :param y_spt:   [support_sz]
        :param x_qry:   [query_sz, c_, h, w]
        :param y_qry:   [query_sz]
        :return:
        """

        support_sz, c_, h, w = x_spt.size()

        assert len(x_spt.shape) == 4

        query_sz = x_qry.size(0)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        shared_net = deepcopy(self.shared_net)
        nway_net = deepcopy(self.nway_net)
        discrim_net = deepcopy(self.discrim_net)
        nets = (shared_net, nway_net, discrim_net)

        loss_q, corrects = self.single_task_forward(x_spt, y_spt, x_qry, y_qry, nets=nets)

        del shared_net
        del nway_net
        del discrim_net
        del nets # this may not be necessary

        accs = corrects / query_sz

        return accs



def main():
    pass


if __name__ == '__main__':
    main()
