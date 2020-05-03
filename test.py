from dataset import Raw2RgbDataset, denormalize
from torch.utils.data import DataLoader
from model import CAUNet
from os import makedirs
import torch
from torch import load
from torchvision.utils import save_image
import argparse
import torch.backends.cudnn as cudnn


def test():
    cudnn.benchmark = True

    # argument parser
    parser = argparse.ArgumentParser(description='AIM')
    parser.add_argument('--raw_dir', type=str, default='/data/raw_to_rgb/TestingPhoneRaw/')
    parser.add_argument('--rgb_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--weights_path', type=str, default=None)
    parser.add_argument('--save_folder', type=str, default='fidelity_result_images/')
    args = parser.parse_args()
    print(args)

    # data loader
    dataset = Raw2RgbDataset(raw_dir=args.raw_dir,
                             rgb_dir=args.rgb_dir)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    # model
    model = CAUNet(in_channels=3)

    # load weights
    model.load_state_dict(
        load('trained_model/p4_2_second.net')['state_dict'])
    model.eval()
    model.cuda()

    model_first = CAUNet()
    model_first.eval()
    model_first.cuda()
    model_first.load_state_dict(load('trained_model/p4_2.net'))

    second_model_front = CAUNet(in_channels=4)
    second_model_last = CAUNet(in_channels=3)
    second_model_front.eval()
    second_model_last.eval()
    second_model_front.cuda()
    second_model_last.cuda()
    second_model_front.load_state_dict(
        load('trained_model/p4_1_hue.net')['state_dict'])
    second_model_last.load_state_dict(
        load('trained_model/p4_1_hue_second.net')[
            'state_dict'])

    third_model_front = CAUNet(in_channels=4)
    third_model_front.eval()
    third_model_front.cuda()
    third_model_front.load_state_dict(
        load('trained_model/p5_1_hue.pkl')['state_dict'])

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

            first_out_front = model_first(raw_images)
            first_out = model(first_out_front)
            first_out = denormalize(first_out)

            second_out_front = second_model_front(raw_images)
            second_out = second_model_last(second_out_front)
            second_out = denormalize(second_out)

            thrid_out_front = third_model_front(raw_images)
            thrid_out = denormalize(thrid_out_front)

            out = (first_out + second_out + thrid_out) / 3

        # save image, compute psnr, ssim
        for i in range(out.size(0)):
            index = batch_idx*args.batch_size + i
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
    test()

