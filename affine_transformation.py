import torch
import airlab as al
import airlab.utils as utils


def get_affine_transformation(moving_image_path, fixed_image_path, iterations=1000, verbose=False):
    dtype = torch.float32
    device = torch.device("cuda")

    fixed_image = utils.Image.read(fixed_image_path, dtype, device)
    moving_image = utils.Image.read(moving_image_path, dtype, device)

    registration = al.PairwiseRegistration(verbose=verbose)
    transformation = al.transformation.pairwise.AffineTransformation(moving_image, opt_cm=True)
    transformation.init_translation(fixed_image)
    registration.set_transformation(transformation)
    image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)
    registration.set_image_loss([image_loss])
    optimizer = torch.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)
    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(iterations)
    registration.start()

    return transformation


def apply_affine_transformation(img, transformation):
    dtype = torch.float32
    device = torch.device("cuda")

    img = al.Image(img).to(dtype, device)
    displacement = transformation.get_displacement()

    img = al.transformation.utils.warp_image(img, displacement).to(torch.float32, device)

    return img.itk()