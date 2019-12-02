import  torch, os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse

from    meta_gan import MetaGAN

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)


    shared_config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
    ]

    nway_config = [
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    discriminator_config = [
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [1, 64]),
        ('sigmoid', [True])
    ]

    gen_config = [
        ('random_proj', [100, 512, 64, 7]), # [latent_dim, emb_size, ch_out, h_out/w_out]
        # img: (64, 7, 7)
        ('convt2d', [64, 32, 4, 4, 2, 1]), # [ch_in, ch_out, kernel_sz, kernel_sz, stride, padding]
        ('bn', [32]),
        ('relu', [True]),
        # img: (32, 14, 14)
        ('convt2d', [32, 1, 4, 4, 2, 1]),
        # img: (1, 28, 28)
        ('sigmoid', [True])
    ]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mamlGAN = MetaGAN(args, shared_config, nway_config, discriminator_config, gen_config).to(device)


    tmp = filter(lambda x: x.requires_grad, mamlGAN.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(mamlGAN)
    print('Total trainable tensors:', num)

    db_train = OmniglotNShot('omniglot',
                       batchsz=args.tasks_per_batch,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       img_sz=args.img_sz)

    for step in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = mamlGAN(x_spt, y_spt, x_qry, y_qry)

        if step % 50 == 0:
            print('step:', step, '\ttraining acc:', accs)

        if step % 500 == 0:
            accs = []
            for _ in range(1000//args.tasks_per_batch):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc, _ = mamlGAN.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc)

            # [b, update_steps+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--img_sz', type=int, help='img_sz', default=28)
    argparser.add_argument('--img_c', type=int, help='img_c', default=1)
    argparser.add_argument('--tasks_per_batch', type=int, help='meta batch size, i.e. number of tasks per batch', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_steps', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_steps_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--learn_inner_lr', type=bool, help='whether to learn the inner update lr', default=True)

    args = argparser.parse_args()

    main(args)
