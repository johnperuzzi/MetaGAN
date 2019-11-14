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
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.shared_net = Learner(shared_config, args.imgc, args.imgsz)
        self.nway_net = Learner(nway_config, args.imgc, args.imgsz)
        self.discrim_net = Learner(discriminator_config, args.imgc, args.imgsz)

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


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):
            # 0. fake gen examples
            x_gen, y_gen = torch.rand_like(x_spt[i]), y_spt[i]

            # 1. run the i-th task and compute loss for k=0

            FloatTensor = torch.FloatTensor #  torch.cuda.FloatTensor if cuda else 
            valid = Variable(FloatTensor(setsz, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(setsz, 1).fill_(0.0), requires_grad=False)

            # run discriminator on real data
            real_shared_layer = self.shared_net(x_spt[i], vars=None, bn_training=True)
            real_class_logits = self.nway_net(real_shared_layer, vars=None, bn_training=True)
            real_valid_preds = self.discrim_net(real_shared_layer, vars=None, bn_training=True)


            real_nway_loss = F.cross_entropy(real_class_logits, y_spt[i])
            real_valid_loss = F.mse_loss(real_valid_preds, valid)

            # run discriminator on generated data
            gen_shared_layer = self.shared_net(x_gen, vars=None, bn_training=True)
            gen_class_logits = self.nway_net(gen_shared_layer, vars=None, bn_training=True)
            gen_valid_preds = self.discrim_net(gen_shared_layer, vars=None, bn_training=True)

            gen_nway_loss = F.cross_entropy(gen_class_logits, y_spt[i]) #real_nway_loss 
            gen_valid_loss = F.mse_loss(gen_valid_preds, fake)


            nway_loss = gen_nway_loss + real_nway_loss
            valid_loss = gen_valid_loss + real_valid_loss

            shared_loss = nway_loss + valid_loss


            # 2. compute grad on theta_pi
            
            n_grad = torch.autograd.grad(nway_loss, self.nway_net.parameters(), retain_graph=True)
            d_grad = torch.autograd.grad(valid_loss, self.discrim_net.parameters(), retain_graph=True)
            s_grad = torch.autograd.grad(shared_loss, self.shared_net.parameters())


            fast_s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(s_grad, self.shared_net.parameters())))
            fast_n_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(n_grad, self.nway_net.parameters())))
            fast_d_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(d_grad, self.discrim_net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                
                q_shared_layer = self.shared_net(x_qry[i], self.shared_net.parameters(), bn_training=True)
                q_class_logits = self.nway_net(q_shared_layer, self.nway_net.parameters(), bn_training=True)
                gen_valid_preds = self.discrim_net(q_shared_layer, self.discrim_net.parameters(), bn_training=True)
                # print(x_qry.shape)
                # print(y_qry.shape)
                # print(y_qry[i].shape)
                # print(q_class_logits.shape)
                loss_q = F.cross_entropy(q_class_logits, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(q_class_logits, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]

                q_shared_layer = self.shared_net(x_qry[i], fast_s_weights, bn_training=True)
                q_class_logits = self.nway_net(q_shared_layer, fast_n_weights, bn_training=True)
                gen_valid_preds = self.discrim_net(q_shared_layer, fast_d_weights, bn_training=True)

                loss_q = F.cross_entropy(q_class_logits, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(q_class_logits, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 0. generate images for each class
                x_gen, y_gen = torch.rand_like(x_spt[i]), y_spt[i]

                FloatTensor = torch.FloatTensor #  torch.cuda.FloatTensor if cuda else 
                valid = Variable(FloatTensor(setsz, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(setsz, 1).fill_(0.0), requires_grad=False)


                # run discriminator on real data
                real_shared_layer = self.shared_net(x_spt[i], fast_s_weights, bn_training=True)
                real_class_logits = self.nway_net(real_shared_layer, fast_n_weights, bn_training=True)
                real_valid_preds = self.discrim_net(real_shared_layer, fast_d_weights, bn_training=True)

                real_nway_loss = F.cross_entropy(real_class_logits, y_spt[i])
                real_valid_loss = F.mse_loss(real_valid_preds, valid)

                # run discriminator on generated data
                gen_shared_layer = self.shared_net(x_gen, fast_s_weights, bn_training=True)
                gen_class_logits = self.nway_net(gen_shared_layer, fast_n_weights, bn_training=True)
                gen_valid_preds = self.discrim_net(gen_shared_layer, fast_d_weights, bn_training=True)

                gen_nway_loss = F.cross_entropy(gen_class_logits, y_spt[i]) # real_nway_loss#
                gen_valid_loss = F.mse_loss(gen_valid_preds, fake)

                nway_loss = gen_nway_loss + real_nway_loss
                valid_loss = gen_valid_loss + real_valid_loss

                shared_loss = nway_loss + valid_loss



                # 2. compute grad on theta_pi
                
                n_grad = torch.autograd.grad(nway_loss, fast_n_weights, retain_graph=True)
                d_grad = torch.autograd.grad(valid_loss, fast_d_weights, retain_graph=True)
                s_grad = torch.autograd.grad(shared_loss, fast_s_weights)

                # 3. theta_pi = theta_pi - train_lr * grad
                fast_s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(s_grad, fast_s_weights)))
                fast_n_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(n_grad, fast_n_weights)))
                fast_d_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(d_grad, fast_d_weights)))




                q_shared_layer = self.shared_net(x_qry[i], fast_s_weights, bn_training=True)
                q_class_logits = self.nway_net(q_shared_layer, fast_n_weights, bn_training=True)

                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(q_class_logits, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(q_class_logits, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num


        # optimize theta parameters
        self.meta_optim.zero_grad()
        # self.meta_shared_optim.zero_grad()
        # self.meta_nway_optim.zero_grad()
        # self.meta_discrim_optim.zero_grad()

        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()
        # self.meta_shared_optim.step()
        # self.meta_nway_optim.step()
        # self.meta_discrim_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        setsz, c_, h, w = x_spt.size()

        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        shared_net = deepcopy(self.shared_net)
        nway_net = deepcopy(self.nway_net)
        discrim_net = deepcopy(self.discrim_net)

        x_gen, y_gen = torch.rand_like(x_spt), y_spt

        FloatTensor = torch.FloatTensor #  torch.cuda.FloatTensor if cuda else 
        valid = Variable(FloatTensor(setsz, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(setsz, 1).fill_(0.0), requires_grad=False)

        # 1. run the i-th task and compute loss for k=0

        # run discriminator on real data
        real_shared_layer = shared_net(x_spt)
        real_class_logits = nway_net(real_shared_layer)
        real_valid_preds = discrim_net(real_shared_layer)

        real_nway_loss = F.cross_entropy(real_class_logits, y_spt)
        real_valid_loss = F.mse_loss(real_valid_preds, valid)

        # run discriminator on generated data
        gen_shared_layer = shared_net(x_gen)
        gen_class_logits = nway_net(gen_shared_layer)
        gen_valid_preds = discrim_net(gen_shared_layer)

        gen_nway_loss = F.cross_entropy(gen_class_logits, y_spt) #real_nway_loss
        gen_valid_loss = F.mse_loss(gen_valid_preds, fake)

        nway_loss = gen_nway_loss + real_nway_loss
        valid_loss = gen_valid_loss + real_valid_loss

        shared_loss = nway_loss + valid_loss

        
        n_grad = torch.autograd.grad(nway_loss, nway_net.parameters(), retain_graph=True)
        d_grad = torch.autograd.grad(valid_loss, discrim_net.parameters(), retain_graph=True)
        s_grad = torch.autograd.grad(shared_loss, shared_net.parameters())


        fast_s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(s_grad, shared_net.parameters())))
        fast_n_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(n_grad, nway_net.parameters())))
        fast_d_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(d_grad, discrim_net.parameters())))





        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            q_shared_layer = shared_net(x_qry, shared_net.parameters(), bn_training=True)
            q_class_logits = nway_net(q_shared_layer, nway_net.parameters(), bn_training=True)
            gen_valid_preds = discrim_net(q_shared_layer, discrim_net.parameters(), bn_training=True)


            # [setsz]
            pred_q = F.softmax(q_class_logits, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            q_shared_layer = shared_net(x_qry, fast_s_weights, bn_training=True)
            q_class_logits = nway_net(q_shared_layer, fast_n_weights, bn_training=True)
            gen_valid_preds = discrim_net(q_shared_layer, fast_d_weights, bn_training=True)


            # [setsz]
            pred_q = F.softmax(q_class_logits, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # # 1. run the i-th task and compute loss for k=1~K-1
            # logits = net(x_spt, fast_weights, bn_training=True)
            # loss = F.cross_entropy(logits, y_spt)

            # # 2. compute grad on theta_pi
            # grad = torch.autograd.grad(loss, fast_weights)
            # # 3. theta_pi = theta_pi - train_lr * grad
            # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))


            x_gen, y_gen = torch.rand_like(x_spt), y_spt
            FloatTensor = torch.FloatTensor #  torch.cuda.FloatTensor if cuda else
            valid = Variable(FloatTensor(setsz, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(setsz, 1).fill_(0.0), requires_grad=False)

            # 1. run the i-th task and compute loss for k=0

            # run discriminator on real data
            real_shared_layer = shared_net(x_spt, fast_s_weights, bn_training=True)
            real_class_logits = nway_net(real_shared_layer, fast_n_weights, bn_training=True)
            real_valid_preds = discrim_net(real_shared_layer, fast_d_weights, bn_training=True)

            real_nway_loss = F.cross_entropy(real_class_logits, y_spt)
            real_valid_loss = F.mse_loss(real_valid_preds, valid)
            # run discriminator on generated data
            gen_shared_layer = shared_net(x_gen, fast_s_weights, bn_training=True)
            gen_class_logits = nway_net(gen_shared_layer, fast_n_weights, bn_training=True)
            gen_valid_preds = discrim_net(gen_shared_layer, fast_d_weights, bn_training=True)

            gen_nway_loss = F.cross_entropy(gen_class_logits, y_spt) # real_nway_loss#
            gen_valid_loss = F.mse_loss(gen_valid_preds, fake)

            nway_loss = gen_nway_loss + real_nway_loss
            valid_loss = gen_valid_loss + real_valid_loss

            shared_loss = nway_loss + valid_loss

            
            n_grad = torch.autograd.grad(nway_loss, fast_n_weights, retain_graph=True)
            d_grad = torch.autograd.grad(valid_loss, fast_d_weights, retain_graph=True)
            s_grad = torch.autograd.grad(shared_loss, fast_s_weights)


            fast_s_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(s_grad,  fast_s_weights)))
            fast_n_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(n_grad, fast_n_weights)))
            fast_d_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(d_grad, fast_d_weights)))



            q_shared_layer = shared_net(x_qry, fast_s_weights, bn_training=True)
            q_class_logits = nway_net(q_shared_layer, fast_n_weights, bn_training=True)



            # logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(q_class_logits, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(q_class_logits, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del shared_net
        del nway_net
        del discrim_net

        accs = np.array(corrects) / querysz

        return accs




def main():
    pass


if __name__ == '__main__':
    main()
