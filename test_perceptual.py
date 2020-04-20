from dataset import Raw2RgbDataset, denormalize
from torch.utils.data import DataLoader
from model import CAUNet
from os import makedirs
import torch
from torch import load
from torch.nn.functional import pad
from torchvision.utils import save_image
import argparse


def test_fullres():
    # argument parser
    parser = argparse.ArgumentParser(description='AIM')
    parser.add_argument('--raw_dir', type=str, default='/data/raw_to_rgb/TestingPhoneRaw/')
    parser.add_argument('--rgb_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--first_weights_path', type=str,
                        default='trained_model/Perceptual_conv4_2/finetune_110/model_0.net')
    parser.add_argument('--second_weights_path', type=str,
                        default='trained_model/Perceptual_conv4_2/finetune_110_two_stage_finetune/model_1.net')
    parser.add_argument('--save_folder', type=str, default='Perceptual_Results/')
    args = parser.parse_args()
    print(args)

    # data loader
    dataset = Raw2RgbDataset(raw_dir=args.raw_dir,
                             rgb_dir=args.rgb_dir)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    # model
    model = CAUNet(in_channels=3)

    model_front = CAUNet(in_channels=4)
    model_last = CAUNet(in_channels=3)
    model_front.eval()
    model_last.eval()
    model_front.cuda()
    model_last.cuda()
    model_front.load_state_dict(
        load(args.first_weights_path))
    model_last.load_state_dict(
        load(args.second_weights_path)['state_dict'])

    # save folder
    save_folder = args.save_folder
    makedirs(save_folder, exist_ok=True)

    # eval metric
    total_psnr = 0
    total_ssim = 0

    # for each batch
    for batch_idx, data in enumerate(data_loader):

        raw_images = data
        raw_images = raw_images.cuda()

        with torch.no_grad():
            out_front = model_front(raw_images)
            out = model_last(out_front)
            out = denormalize(out)

        for i in range(out.size(0)):
            index = batch_idx * args.batch_size + i
            image_path = save_folder + str(index) + '.png'
            save_image(tensor=out[i], filename=image_path)

            message = 'id: {}'.format(index)
            print(message)

    # average
    total_psnr /= len(dataset)
    total_ssim /= len(dataset)

    # Print psnr, ssim
    message = '\t {}: {:.2f}\t {}: {:.4f}'.format('psnr', total_psnr, 'ssim', total_ssim)

    print(message)


if __name__ == '__main__':
    test_fullres()

