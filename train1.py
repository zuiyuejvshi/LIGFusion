import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from net import Restormer_Encoder, Restormer_Decoder
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, cc
import os
import sys
import time
import datetime
import kornia

def main():

    criteria_fusion = Fusionloss()
    model_str = 'LIGFusion'


    num_epochs = 40

    lr = 1e-4
    weight_decay = 0
    batch_size = 32
    GPU_number = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

    coeff_mse_loss_VF = 1.0
    coeff_mse_loss_IF = 1.0
    coeff_decomp = 2.0
    coeff_tv = 5.0

    clip_grad_norm_value = 0.01
    optim_step = 20
    optim_gamma = 0.5


    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    DIDF_Encoder = Restormer_Encoder().to(device)
    DIDF_Decoder = Restormer_Decoder().to(device)

    DIDF_Encoder = nn.DataParallel(DIDF_Encoder)
    DIDF_Decoder = nn.DataParallel(DIDF_Decoder)

    optimizer1 = optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer2 = optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)

    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    Loss_ssim = kornia.losses.SSIM(11, reduction='mean')


    trainloader = DataLoader(
        H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    loader = {'train': trainloader}
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    best_loss = float('inf')
    best_epoch = 0


    step = 0
    torch.backends.cudnn.benchmark = True
    prev_time = time.time()

    for epoch in range(num_epochs):
        for i, (data_VIS, data_IR, _) in enumerate(loader['train']):
            data_VIS = data_VIS.to(device, non_blocking=True)
            data_IR = data_IR.to(device, non_blocking=True)

            DIDF_Encoder.train()
            DIDF_Decoder.train()

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS, is_vis=False, training_phase=1)[:3]
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR, is_vis=False, training_phase=1)[:3]
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()


            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch + 1
                checkpoint = {
                    'epoch': epoch + 1,
                    'DIDF_Encoder': DIDF_Encoder.state_dict(),
                    'DIDF_Decoder': DIDF_Decoder.state_dict(),
                    'BaseFeatureExtraction': DIDF_Encoder.module.baseFeature.state_dict(),
                    'DetailFeatureExtraction': DIDF_Encoder.module.detailFeature.state_dict(),
                    'optimizer1': optimizer1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),
                    'scheduler1': scheduler1.state_dict(),
                    'scheduler2': scheduler2.state_dict(),
                }
                torch.save(checkpoint, os.path.join("models/best_phase1.pth"))

            batches_done = epoch * len(loader['train']) + i
            batches_left = num_epochs * len(loader['train']) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %s"
                % (
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    len(loader['train']),
                    loss.item(),
                    str(time_left).split('.')[0],
                )
            )
            sys.stdout.flush()

        print()

        scheduler1.step()
        scheduler2.step()


        for optimizer in [optimizer1, optimizer2]:
            for param_group in optimizer.param_groups:
                if param_group['lr'] <= 1e-6:
                    param_group['lr'] = 1e-6

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'DIDF_Encoder': DIDF_Encoder.state_dict(),
                'DIDF_Decoder': DIDF_Decoder.state_dict(),
                'BaseFeatureExtraction': DIDF_Encoder.module.baseFeature.state_dict(),
                'DetailFeatureExtraction': DIDF_Encoder.module.detailFeature.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'scheduler1': scheduler1.state_dict(),
                'scheduler2': scheduler2.state_dict(),
            }
            torch.save(checkpoint, os.path.join("models/LIGFuse_" + timestamp + f'_epoch_{epoch + 1}.pth'))

    print("Phase I Training completed.")

if __name__ == '__main__':
    main()
