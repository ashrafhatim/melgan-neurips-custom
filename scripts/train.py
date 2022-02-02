import sys
sys.path.insert(0,'/home/jupyter/melgan-neurips-custom')

from mel2wav.dataset import AudioDataset
from mel2wav.modules import Generator, Discriminator, Audio2Mel
from mel2wav.utils import save_sample, mel_rec_val_loss
from mel2wav.custom_transforms import change_speed, change_amplitude

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import yaml
import numpy as np
import time
import argparse
from pathlib import Path

import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--load_path", default=None)

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3) # new
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)
    parser.add_argument("--cond_disc", action="store_true")

    parser.add_argument("--data_path", default=None, type=Path)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=list, default=[8192, 1024, 512])

    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--n_test_samples", type=int, default=8)
    
    # parser.add_argument("--augment", type=bool, default=True)# dont change this, change the code in dataset class if you want to add augmentation
    parser.add_argument("--save_checkpoints", type=bool, default=True)
    parser.add_argument("--load_from_checkpoints", type=bool, default=True)
    # parser.add_argument("--steps", type=int, default=0)
    
    parser.add_argument("--pre_trained", type=bool, default=False)
    
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # print("agument: ", args.augment)
    print("gpu_id: ", args.gpu_id)
    print("num_D: ", args.num_D)
    print("save_checkpoints: ", args.save_checkpoints)
    print("load_from_checkpoints: ", args.load_from_checkpoints)
    print("pre_trained: ", args.pre_trained)

    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None
    root.mkdir(parents=True, exist_ok=True)
    
    # make sure to not overwrite the tensorboard writer
    assert (os.path.exists(root /  "steps.pt" ) and args.load_path) or not os.path.exists(root /  "steps.pt" ), "Forgot to provide the load_path !!, make sure to add it to prevent overriding tensorboard."


    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    #######################
    # Load PyTorch Models #
    #######################
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).cuda(args.gpu_id)
    if args.pre_trained:
        netG.load_state_dict(torch.load("/home/jupyter/melgan-neurips-custom/models/linda_johnson.pt"))
        print("the Generator pre-trained model has been loaded completely!")

        
    netD = Discriminator(
        args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor
    ).cuda(args.gpu_id)
    if args.pre_trained:
        netD.load_state_dict(torch.load("/home/jupyter/melgan-neurips-custom/logs/exp7/best_netD.pt"))
        print("the Discriminator pre-trained model has been loaded completely!")


    netD_helper = Discriminator(
        1, args.ndf//2, args.n_layers_D, args.downsamp_factor, 512
    ).cuda(args.gpu_id)    
    fft = Audio2Mel(n_mel_channels=args.n_mel_channels).cuda(args.gpu_id)

    # print(netG)
    # print(netD)

    #####################
    # Create optimizers #
    #####################
    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam([{"params":netD.parameters()}, {"params":netD_helper.parameters()}], lr=1e-4, betas=(0.5, 0.9))
    # optD_helper = torch.optim.Adam(netD_helper.parameters(), lr=1e-4, betas=(0.5, 0.9))

    #######################
    # Create data loaders #
    #######################

#     transform = transforms.RandomChoice(
#     [change_speed([0.99, 1.01], 0.001),
#      change_amplitude(low=0.3, high=1.0),
#      torch.nn.Identity()],
#      p=[2,1,1]
# )
    transform = transforms.RandomChoice(
    [
     change_amplitude(low=0.3, high=1.0)
    ],
     p=[1]
)

    train_set = AudioDataset(
        Path(args.data_path) / "train_files.txt", args.seq_len, sampling_rate=22050, transform=transform, 
        training_files_english= Path(args.data_path) / "train_files_english.txt",
    )
    val_set = AudioDataset(
        Path(args.data_path) / "val_files.txt",
        [22050 * 4],
        sampling_rate=22050,
        augment=False,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1)
    
    print("# of train samples: ", len(train_loader) * args.batch_size)
    print("# of val samples: ", len(val_loader))
    
    # # train the discriminator a bit
    # if args.pre_trained:
    #     step_d = 0
    #     while step_d < 2000:
    #         for iterno, x_t in enumerate(train_loader):
    #             x_t = [ele.cuda(args.gpu_id) for ele in x_t]

    #             s_t = fft(x_t[0]).detach()
    #             s_t1 = fft(x_t[1]).detach()

    #             x_pred_t = netG(s_t.cuda(args.gpu_id))
    #             x_pred_t1 = netG(s_t1.cuda(args.gpu_id))

    #             with torch.no_grad():
    #                 s_pred_t = fft(x_pred_t.detach())
    #                 s_pred_t1 = fft(x_pred_t1.detach())
    #                 s_error = F.l1_loss(s_t, s_pred_t).item()
    #                 s_error += F.l1_loss(s_t1, s_pred_t1).item()

    #             # Train Discriminator #
                
    #             # original
    #             D_fake_det = netD(x_pred_t.cuda(args.gpu_id).detach())
    #             D_real = netD(x_t[0].cuda(args.gpu_id))

    #             loss_D = 0
    #             for scale in D_fake_det:
    #                 loss_D += F.relu(1 + scale[-1]).mean()

    #             for scale in D_real:
    #                 loss_D += F.relu(1 - scale[-1]).mean()

    #             # new
    #             D_fake_det1 = netD_helper(x_pred_t1.cuda(args.gpu_id).detach())
    #             D_real1 = netD_helper(x_t[1].cuda(args.gpu_id))

    #             # loss_D = 0
    #             for scale in D_fake_det1:
    #                 loss_D += F.relu(1 + scale[-1]).mean()

    #             for scale in D_real1:
    #                 loss_D += F.relu(1 - scale[-1]).mean()

    #             netD.zero_grad()
    #             netD_helper.zero_grad()
    #             loss_D.backward()
    #             optD.step()
    #             # optD_helper.step()

    #             step_d += 1
    #             if step_d >= 2000:
    #                 break
                    

    #############################################
    # Load models & calculate the offset epochs #
    #############################################

    steps = 0
    epoch_offset = 0
    if load_root and load_root.exists():
        steps = torch.load(load_root / "steps.pt")

        if args.load_from_checkpoints:
            # steps = args.steps
            print(load_root / ("netG_%d.pt" % steps))
            netG.load_state_dict(torch.load(load_root / ("netG_%d.pt" % steps), map_location='cuda:%d' % args.gpu_id))
            optG.load_state_dict(torch.load(load_root / ("optG_%d.pt" % steps), map_location='cuda:%d' % args.gpu_id))
            netD.load_state_dict(torch.load(load_root / ("netD_%d.pt" % steps), map_location='cuda:%d' % args.gpu_id))
            optD.load_state_dict(torch.load(load_root / ("optD_%d.pt" % steps), map_location='cuda:%d' % args.gpu_id))

            netD_helper.load_state_dict(torch.load(load_root / ("netD_helper_%d.pt" % steps), map_location='cuda:%d' % args.gpu_id))
            # optD_helper.load_state_dict(torch.load(load_root / ("optD_helper_%d.pt" % steps), map_location='cuda:%d' % args.gpu_id))
        else:
            netG.load_state_dict(torch.load(load_root / ("netG.pt" ), map_location='cuda:%d' % args.gpu_id))
            optG.load_state_dict(torch.load(load_root / ("optG.pt" ), map_location='cuda:%d' % args.gpu_id))
            netD.load_state_dict(torch.load(load_root / ("netD.pt" ), map_location='cuda:%d' % args.gpu_id))
            optD.load_state_dict(torch.load(load_root / ("optD.pt" ), map_location='cuda:%d' % args.gpu_id))

            netD_helper.load_state_dict(torch.load(load_root / ("netD_helper.pt" ), map_location='cuda:%d' % args.gpu_id))
            # optD_helper.load_state_dict(torch.load(load_root / ("optD_helper.pt" ), map_location='cuda:%d' % args.gpu_id))
        
        steps = steps + 1
        epoch_offset = max(0, int(steps / len(train_loader)))

    ##########################
    # Dumping original audio #
    ##########################
    val_voc = []
    val_audio = []
    for i, x_t in enumerate(val_loader):
        x_t = x_t.cuda(args.gpu_id)
        s_t = fft(x_t).detach()

        val_voc.append(s_t.cuda(args.gpu_id))
        val_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), 22050, audio)
        writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=22050)

        if i == args.n_test_samples - 1:
            break

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000000
    for epoch in range(epoch_offset + 1, args.epochs + 1):
        for iterno, x_t in enumerate(train_loader):
            x_t = [ele.cuda(args.gpu_id) for ele in x_t]

            # rearrange the effective batch
            if x_t[1].shape[1] == 2:
                t0 = x_t[1][:,0:1,:]
                t1 = x_t[1][:,1:2,:]
                x_t[1] = torch.concat((t0,t1), axis=0)


            s_t = fft(x_t[0]).detach()
            x_pred_t = netG(s_t.cuda(args.gpu_id))

            s_t1 = fft(x_t[1]).detach()
            x_pred_t1 = netG(s_t1.cuda(args.gpu_id))

            # sample 512 window
            max_audio_start = args.seq_len[1] - args.seq_len[2]
            audio_start = random.randint(0, max_audio_start)
            
            x_pred_t1 = x_pred_t1[:,:,audio_start : audio_start + args.seq_len[2]]
            x_t[1] = x_t[1][:,:,audio_start : audio_start + args.seq_len[2]]
            s_t1 = fft(x_t[1]).detach()


            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_pred_t1 = fft(x_pred_t1.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()
                s_error += F.l1_loss(s_t1, s_pred_t1).item()

            #######################
            # Train Discriminator #
            #######################
            # original
            D_fake_det = netD(x_pred_t.cuda(args.gpu_id).detach())
            D_real = netD(x_t[0].cuda(args.gpu_id))

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            # new
            D_fake_det1 = netD_helper(x_pred_t1.cuda(args.gpu_id).detach())
            D_real1 = netD_helper(x_t[1].cuda(args.gpu_id))

            # loss_D = 0
            for scale in D_fake_det1:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real1:
                loss_D += F.relu(1 - scale[-1]).mean()

            netD.zero_grad()
            netD_helper.zero_grad()
            loss_D.backward()
            optD.step()
            # optD_helper.step()

            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t.cuda(args.gpu_id))
            D_fake1 = netD_helper(x_pred_t1.cuda(args.gpu_id))

            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()
            for scale in D_fake1:
                loss_G += -scale[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (args.n_layers_D + 1)
            D_weights = 1.0 / args.num_D
            wt = D_weights * feat_weights
            for i in range(args.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())
            
            for j in range(len(D_fake1[0]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake1[0][j], D_real1[0][j].detach())

            netG.zero_grad()
            (loss_G + args.lambda_feat * loss_feat).backward()
            optG.step()

            ######################
            # Update tensorboard #
            ######################
            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error])

            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
            steps += 1

            if steps % args.save_interval == 0:
                st = time.time()
                with torch.no_grad():
                    for i, (voc, _) in enumerate(zip(val_voc, val_audio)):
                        pred_audio = netG(voc)
                        pred_audio = pred_audio.squeeze().cpu()
                        save_sample(root / ("generated_%d.wav" % i), 22050, pred_audio)
                        writer.add_audio(
                            "generated/sample_%d.wav" % i,
                            pred_audio,
                            epoch,
                            sample_rate=22050,
                        )

                if args.save_checkpoints:
                    torch.save(netG.state_dict(), root / ("netG_%d.pt" % steps))
                    torch.save(optG.state_dict(), root / ("optG_%d.pt" % steps))

                    torch.save(netD.state_dict(), root / ("netD_%d.pt" % steps))
                    torch.save(optD.state_dict(), root / ("optD_%d.pt" % steps))

                    torch.save(netD_helper.state_dict(), root / ("netD_helper_%d.pt" % steps))
                    # torch.save(optD_helper.state_dict(), root / ("optD_helper_%d.pt" % steps))
                else:
                    torch.save(netG.state_dict(), root / ("netG.pt" ))
                    torch.save(optG.state_dict(), root / ("optG.pt" ))

                    torch.save(netD.state_dict(), root / ("netD.pt" ))
                    torch.save(optD.state_dict(), root / ("optD.pt" ))

                    torch.save(netD_helper.state_dict(), root / ("netD_helper.pt" ))
                    # torch.save(optD_helper.state_dict(), root / ("optD_helper.pt" ))
                    
                
                torch.save(steps, root / "steps.pt")

                mel_reconst = mel_rec_val_loss(val_loader, netG, fft, args.gpu_id)
                writer.add_scalar("loss/val_mel_reconst", mel_reconst, steps)

                # if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                if mel_reconst < best_mel_reconst:
                    best_mel_reconst = mel_reconst
                    torch.save(netD.state_dict(), root / "best_netD.pt")
                    torch.save(netG.state_dict(), root / "best_netG.pt")

                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)

            if steps % args.log_interval == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time.time() - start) / args.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()


if __name__ == "__main__":
    main()
