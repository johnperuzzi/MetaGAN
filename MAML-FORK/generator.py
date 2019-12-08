import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np



class Generator(nn.Module):
    """

    """

    def __init__(self, config, img_c, img_sz, num_classes):
        """

        :param config: network config file, type:list of (string, list)
        :param img_c: 1 or 3
        :param img_sz:  28 or 84
        """
        super(Generator, self).__init__()


        self.config = config

        self.num_classes = num_classes

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernel_sz, kernel_sz, stride, padding]
                # output will be sz = stride * (input_sz) + kernel_sz
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name is "random_proj":
                # [ch_in, ch_out, img_sz]
                latent_dim, latent_ch_out, emb_dim, emb_ch_out, hw_out = param

                # latent projection params
                w_lat = nn.Parameter(torch.ones(hw_out*hw_out*latent_ch_out, latent_dim))
                torch.nn.init.kaiming_normal_(w_lat)

                self.vars.append(w_lat)
                self.vars.append(nn.Parameter(torch.zeros(hw_out*hw_out*latent_ch_out)))

                # embedding projection params
                w_emb = nn.Parameter(torch.ones(hw_out*hw_out*emb_ch_out, emb_dim))
                torch.nn.init.kaiming_normal_(w_emb)

                self.vars.append(w_emb)
                self.vars.append(nn.Parameter(torch.zeros(hw_out*hw_out*emb_ch_out)))


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError


    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'

            elif name is 'random_proj':
                info += 'random_proj:(hidden_sz:%d, embedding_size:%d, height_width:%d, ch_out:%d)'%(param[0], param[1], param[2], param[3]) + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, y, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 512]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        batch_sz = x.size()[0]

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        assert self.config[0][0] is 'random_proj'
        # need to start with the random projection
        for name, param in self.config:
            # print(name)
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'random_proj':
                # copying generator architecture from here: https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

                latent_dim, latent_ch_out, emb_dim, emb_ch_out, hw_out = param
                cuda = torch.cuda.is_available()

                # send random tensor to linear layer, reshape into noise channels
                FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor 
                rand = FloatTensor((batch_sz, latent_dim))
                torch.randn((batch_sz, latent_dim), out=rand, requires_grad=True)
                w_lat, b_lat = vars[idx], vars[idx + 1]
                rand = F.linear(rand, w_lat, b_lat)
                rand = F.leaky_relu(rand, 0.2)
                rand = rand.view(rand.size(0), latent_ch_out, hw_out, hw_out)

                # send class embbeddings through a linear layer, reshape embeddings channels
                w_emb, b_emb = vars[idx+2], vars[idx + 3]
                x = F.linear(x, w_emb, b_emb)
                x = F.leaky_relu(x, 0.2)
                x = x.view(x.size(0), emb_ch_out, hw_out, hw_out)

                # concatenate embeddings and projections
                x = torch.cat((x, rand), 1)
                idx += 4



            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        # right now still returning y so that we can easilly extend to generating diff nums of examples by adjusting y in here
        return x, y


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars