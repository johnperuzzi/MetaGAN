from    torch import nn
from generator import Generator
from learnear import Learner

class Full_Net(nn.Module):
    def __init__(self, gen_config, shared_config, discriminator_config, nway_config, img_c, img_sz, n_way):
        super(Full_Net, self).__init__

        # self.vars = nn.ParameterList()
        # self.vars_bn = nn.ParameterList()

        self.generator = Generator(gen_config, img_c, img_sz, n_way)
        self.shared_net = Learner(shared_config, args.img_c, args.img_sz)
        self.discrim_net = Learner(discriminator_config, args.img_c, args.img_sz)
        self.nway_net = Learner(nway_config, args.img_c, args.img_sz)
        self.nets = (self.generator, self.shared_net, self.discrim_net, self.nway_net)


    def forward(self, x, y, conditions=None, bn_training=True, nway=True, discrim=True):
        gen_x, gen_y = self.generator(x, y, bn_training=bn_training)
        gen_class_logits, gen_discrim_logits = self.pred(gen_x, nway=nway, discrim=discrim, bn_training=bn_training) # not sure what to set conditions to
        real_class_logits, real_discrim_logits = self.pred(x, nway=nway, discrim=discrim, bn_training=bn_training, conditions=conditions)

        return gen_class_logits, gen_discrim_logits, real_class_logits, real_discrim_logits

    # Returns predicted class logits and descriminator outputs according
    # to the input "x", using the shared/nway/discrim nets and weights provided
    # pass in 'conditions' if using conditioned discriminator
    def pred(self, x, nway=True, discrim=True, conditions=None, bn_training=True):

        shared_layer = self.shared_net(x, bn_training=bn_training)
        discrim_logits = self.discrim_net(shared_layer, conditions=conditions, bn_training=bn_training) if discrim else None
        class_logits = self.nway_net(shared_layer, bn_training=bn_training) if nway else None
          
        return class_logits, discrim_logits


    def zero_grad(self, vars=None):
        for net in self.nets:
            net.zero_grad()

    # def parameters(self):
    #     """
    #     override this function since initial parameters will return with a generator.
    #     :return:
    #     """
    #     return self.vars
