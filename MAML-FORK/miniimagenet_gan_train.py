import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from matplotlib import pyplot as plt
from PIL import Image
import json
from datetime import datetime
from meta_gan import MetaGAN


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        print("you probably tried to make a new model in the same minute, wait a couple seconds")
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def save_accs(path, accs):
    file = open(path +  '/q_nway_accuracies.txt', 'ab')
    np.savetxt(file, np.array([accs["q_nway"]]))
    file.close()

    file = open(path +  '/q_discrim_accuracies.txt', 'ab')
    np.savetxt(file, np.array([accs["q_discrim"]]))
    file.close()

    file = open(path +  '/gen_nway_accuracies.txt', 'ab')
    np.savetxt(file, np.array([accs["gen_nway"]]))
    file.close()

    file = open(path +  '/gen_discrim_accuracies.txt', 'ab')
    np.savetxt(file, np.array([accs["gen_discrim"]]))
    file.close()


def save_imgs(path, imgs, step):
    # save raw txt files
    img_f=open(path+"/images_step" + str(step) + ".txt",'ab')
    some_imgs = np.reshape(imgs, [imgs.shape[0]*imgs.shape[1], -1])[0:50]
    np.savetxt(img_f,some_imgs)
    img_f.close()

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # save png of imgs
    i = 0
    for flat_img in some_imgs:
        img = flat_img.reshape(3,84,84).swapaxes(0,1).swapaxes(1,2)
        im = ((img - np.min(img))*255/(np.max(img - np.min(img)))).astype(np.uint8)
        if i < 49:
            plt.subplot(7, 7, 1 + i)
            plt.axis('off')
            plt.imshow(im)
        i += 1
    plt.savefig(path+"/images_step" + str(step) + ".png")
    plt.close()



def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    shared_config = [
        # [ch_out, ch_in, kernelsz, kernelsz, stride, padding]
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),

    ]

    nway_config = [
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    discriminator_config = [
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [1, 32 * 5 * 5]),
        ('sigmoid', [True])
    ]

    if args.condition_discrim:
        discriminator_config = [
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('leakyrelu', [.2, True]),
            ('bn', [32]),
            ('max_pool2d', [2, 1, 0]),
            ('flatten', []),
            ('condition', [1024, 32 * 5 * 5, 5]),
            ('leakyrelu', [0.2, True]),
            ('bn', [1024]),
            ('linear', [1, 1024]),
            ('sigmoid', [True])
        ]

    gen_config = [
        ('random_proj', [100, 512, 32, 21]), # [latent_dim, embedding_dim, ch_out, h_out/w_out]
        # img: (32, 21, 21)
        ('convt2d', [32, 16, 4, 4, 2, 1]), # [ch_in, ch_out, kernel_sz, kernel_sz, stride, padding]
        ('bn', [16]),
        ('relu', [True]),
        # img: (16, 42, 42)
        ('convt2d', [16, 3, 4, 4, 2, 1]),
        # img: (3, 84, 84)
        ('sigmoid', [True])
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mamlGAN = MetaGAN(args, shared_config, nway_config, discriminator_config, gen_config).to(device)

    tmp = filter(lambda x: x.requires_grad, mamlGAN.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(mamlGAN)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('./data/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.img_sz)
    mini_test = MiniImagenet('./data/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.img_sz)
    

    save_model = not args.no_save
    if save_model:
        now = datetime.now().replace(second=0, microsecond=0)
        path = "results/" + str(now)
        mkdir_p(path)
        file = open(path +  '/architecture.txt', 'w+')
        file.write("shared_config = " + json.dumps(shared_config) + "\n" + 
            "nway_config = " + json.dumps(nway_config) + "\n" +
            "discriminator_config = " + json.dumps(discriminator_config) + "\n" + 
            "gen_config = " + json.dumps(gen_config)
            )
        file.close()

    
    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.tasks_per_batch, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = mamlGAN(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                print("step " + str(step))
                for key in accs.keys():
                    print(key + ": " + str(accs[key]))
                if save_model:
                    save_accs(path, accs)

            if step % 500 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []
                imgs_all_test = []
                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs, imgs = mamlGAN.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)
                    imgs_all_test.append(imgs.cpu().detach().numpy())

                imgs_all_test = np.array(imgs_all_test)

                if save_model:
                    save_imgs(path, imgs_all_test, step)
                
                print('Test acc:', accs)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--img_sz', type=int, help='img_sz', default=84)
    argparser.add_argument('--img_c', type=int, help='img_c', default=3)
    argparser.add_argument('--tasks_per_batch', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_steps', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_steps_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--no_save', default=False, action='store_true', help='Bool type. Pass to not save (right now we save by default)')
    argparser.add_argument('--learn_inner_lr', default=False, action='store_true', help='Bool type. Pass to learn inner lr')
    argparser.add_argument('--condition_discrim', default=False, action='store_true', help='Bool type. Pass to remove n_way loss from generator and condition discriminator')

    args = argparser.parse_args()

    main()
