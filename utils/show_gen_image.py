from matplotlib import pyplot as plt
import os

def show_gen_image(training_folder_name, cap_image, depth_image, gen_image, iters):

    fig = plt.figure()
    plt.axis('off')
    plt.ioff()
    # ax.set_title('sample{}'.format(path))
    fig.add_subplot(2, 4, 1)
    plt.imshow((cap_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2)
    fig.add_subplot(2, 4, 2)
    plt.imshow((cap_image[0, 1, :, :].detach().cpu().numpy() + 1) / 2)
    fig.add_subplot(2, 4, 3)
    plt.imshow((cap_image[0, 2, :, :].detach().cpu().numpy() + 1) / 2)
    fig.add_subplot(2, 4, 4)
    plt.imshow((cap_image[0, 3, :, :].detach().cpu().numpy() + 1) / 2)

    # retrieve distance from normarized depth map
    # 30x10^(-12): [time per frame] x 1024: [frame] x 3x10^8: [speed of light] x depth_map / 2
    # 3 x 1.024 x 3 x depth_map / 2
    fig.add_subplot(2, 4, 5)
    depth_img = (depth_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    depth_img = 3 * 3 * 1.024 * depth_img / 2
    plt.imshow(depth_img)
    fig.add_subplot(2, 4, 6)
    gen_img = (gen_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    gen_img = 3 * 3 * 1.024 * gen_img / 2
    plt.imshow(gen_img)

    fig.add_subplot(2, 4, 7)
    plt.plot(depth_img[128//2, :], label="depth")
    plt.plot(gen_img[128//2, :], label="generate")
    plt.legend(bbox_to_anchor=(1, 1))
    fig.savefig(os.path.join(training_folder_name, 'gen_image_{}.png'.format(iters)))
    plt.close()