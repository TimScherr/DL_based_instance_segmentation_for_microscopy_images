import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(loss_function, label_type):
    """ Get loss function(s) for the training process.

    :param loss_function: Loss function to use.
        :type loss_function: str
    :param label_type: Label type of the training data / predictions, e.g., 'boundary'
        :type label_type: str
    :return: Loss function / dict of loss functions.
    """

    if label_type in ['distance', 'adapted_border']:
        if loss_function == 'l1':
            border_criterion = nn.L1Loss()
            cell_criterion = nn.L1Loss()
        elif loss_function == 'l2':
            border_criterion = nn.MSELoss()
            cell_criterion = nn.MSELoss()
        elif loss_function == 'smooth_l1':
            border_criterion = nn.SmoothL1Loss()
            cell_criterion = nn.SmoothL1Loss()
        elif loss_function == 'bce+ce':
            cell_criterion = bce_dice
            border_criterion = ce_dice
        else:
            raise Exception('Loss unknown')
        criterion = {'border': border_criterion, 'cell': cell_criterion}
    else:
        if loss_function == 'l1':
            criterion = nn.L1Loss()
        elif loss_function == 'l2':
            criterion = nn.MSELoss()
        elif loss_function == 'smooth_l1':
            criterion = nn.SmoothL1Loss()
        elif loss_function == 'ce_dice':
            criterion = ce_dice
        elif loss_function == 'ce':
            criterion = nn.CrossEntropyLoss()
        elif loss_function == 'j_reg_loss':
            criterion = j_regularization_loss
        else:
            raise Exception('Loss unknown')

    return criterion


def dice_loss(y_pred, y_true, use_sigmoid=True):
    """Dice loss: harmonic mean of precision and recall (FPs and FNs are weighted equally). Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, height, width].
        :type y_true:
    :param use_sigmoid: Apply sigmoid activation function to the prediction y_pred.
        :type use_sigmoid: bool
    :return:
    """

    # Avoid division by zero
    smooth = 1.

    # Flatten ground truth
    gt = y_true.contiguous().view(-1)

    if use_sigmoid:  # Apply sigmoid activation to prediction and flatten prediction
        pred = torch.sigmoid(y_pred)
        pred = pred.contiguous().view(-1)
    else:
        pred = y_pred.contiguous().view(-1)

    # Calculate Dice loss
    pred_gt = torch.sum(gt * pred)
    loss = 1 - (2. * pred_gt + smooth) / (torch.sum(gt ** 2) + torch.sum(pred ** 2) + smooth)

    return loss


def bce_dice(y_pred, y_true):
    """ Sum of binary crossentropy loss and Dice loss. Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, height, width].
        :type y_true:
    :return:
    """
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(y_pred, y_true) + dice_loss(y_pred, y_true)

    return loss


def ce_dice(y_pred, y_true, num_classes=3):
    """Sum of crossentropy loss and channel-wise Dice loss.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, height, width].
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    y_true_one_hot = nn.functional.one_hot(y_true, num_classes).float()
    y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2)
    y_pred_softmax = F.softmax(y_pred, dim=1)
    dice_score = 0

    # Crossentropy Loss
    loss_ce = nn.CrossEntropyLoss()
    ce_loss = loss_ce(y_pred, y_true)

    # Channel-wise Dice loss
    for index in range(1, num_classes):
        dice_score += index * dice_loss(y_pred_softmax[:, index, :, :], y_true_one_hot[:, index, :, :],
                                        use_sigmoid=False)

    return ce_loss + 0.5 * dice_score


def j_regularization_loss(y_pred, y_true, num_classes=4, class_weight=0.5):
    """Sum of crossentropy loss and J regularization loss.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, height, width].
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    # Get One-hot-encoding for the j-regularization part of the loss
    y_true_one_hot = nn.functional.one_hot(y_true, num_classes).float()
    y_pred_softmax = F.softmax(y_pred, dim=1)

    # shape y_true: [batchsize, (d), h, w]
    if len(y_true.size()) == 3:  # batch of 2d images
        n_dims = (2, 3)
        y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2)
    elif len(y_true.size()) == 4:  # batch of 3d images
        n_dims = (2, 3, 4)
        y_true_one_hot = y_true_one_hot.permute(0, 4, 1, 2, 3)
    else:
        raise AssertionError(f'unknown tensor shape {y_true.size}')
    dummy_dims = [1 for _ in n_dims]

    # Number of pixels per class. shape: [batch_size, n_classes]
    n_pixels_class = torch.sum(y_true_one_hot, dim=n_dims)

    #########
    # J loss
    #########

    # phi_i : normalize y by number of pixels belonging to class
    # shape phi: [batch_size, n_classes, (d), h, w]
    # add epsilon to avoid nan issues if class label not existent
    phi = y_true_one_hot / (n_pixels_class.reshape((*n_pixels_class.shape, *dummy_dims)) + 1e-12)

    # alpha_i = sum prediction_i*phi_i (sum over all pixels)
    alpha = y_pred_softmax * phi
    # shape alpha: [batchsize, n_classes]
    alpha = torch.sum(alpha, dim=n_dims)

    # beta_ik = sum (1-prediction_i)*phi_k (sum over all pixels)
    inv_pred = 1 - y_pred_softmax
    dummy_vec = torch.arange(y_true_one_hot.size()[1]).to(y_true_one_hot.device)
    # beta :[batchsize, n_classes-1, n_classes]
    beta = torch.stack([torch.sum(phi[:, i != dummy_vec, ...] * inv_pred[:, i, ...].unsqueeze(dim=1), dim=n_dims)
                        for i in range(y_true_one_hot.size()[1])
                        ], dim=-1)
    adding_terms = beta + alpha.reshape((*alpha.size(), 1)).transpose(1, 2) + 1e-12

    lambda_ik = class_weight
    j_loss = -1 * torch.sum(lambda_ik * torch.log(adding_terms / 2))

    ############
    # CE loss
    ############
    # Crossentropy Loss
    loss_ce = nn.CrossEntropyLoss()
    ce_loss = loss_ce(y_pred, y_true)

    # ce_loss is mean of the batch --> divide by batch size. In the paper the J_reg loss part is smaller --> weight
    loss = ce_loss + 0.006 * j_loss / y_true.size()[0]

    # ce_loss2 = -1 / torch.sum(n_pixels_class[0]) * torch.sum(y_true_one_hot * torch.log(y_pred_softmax + 1e-12))

    # loss = ce_loss2 + j_loss

    return loss
