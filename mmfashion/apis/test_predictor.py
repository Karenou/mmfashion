from __future__ import division

from mmcv.parallel import MMDataParallel

from ..core import AttrPredictor, AttrCalculator, CateCalculator
from ..datasets import build_dataloader
from .env import get_root_logger
import numpy as np
import pandas as pd


def test_predictor(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    """ test attribute prediction
    :return top-k accuracy """
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start testing predictor
    if distributed:  # to do
        _dist_test_attr(model, dataset, cfg, validate=validate)
    else:
        _non_dist_test_attr(model, dataset, cfg, validate=validate)


def test_cate_attr_predictor(model,
                             dataset,
                             cfg,
                             distributed=False,
                             validate=False,
                             logger=None):
    """test category and attribute prediction
    :return top-k attribute_accuracy,
            top-k category recall rate """
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start testing predictor
    if distributed:  # to do
        _dist_test_cate_attr(model, dataset, cfg, validate=validate)
    else:
        _non_dist_test_cate_attr(model, dataset, cfg, validate=validate)


def _non_dist_test_attr(model, dataset, cfg, validate=False):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    print('dataloader built')

    model = MMDataParallel(model, device_ids=cfg.gpus.test).cuda()
    model.eval()

    attr_predictor = AttrPredictor(cfg.data.test)

    img_ids, prob_preds = None, None

    for batch_idx, testdata in enumerate(data_loader):
        if batch_idx % cfg.print_interval == 0:
            print("batch %d" % batch_idx)

        img_id = testdata["img_id"].data.cpu().numpy()
        imgs = testdata['img']
        landmark = testdata['landmark']

        attr_pred = model(imgs, attr=None, landmark=landmark, return_loss=False)
        prob_pred = attr_predictor.show_prediction(img_id, attr_pred)

        if not img_ids:
            img_ids = img_id
            prob_preds = prob_pred
        else:
            img_ids = np.concatenate([img_ids, img_id], axis=0)
            prob_preds = np.concatenate([prob_preds, prob_pred], axis=0)

        if batch_idx % cfg.save_interval == 0:
            print("save intermediate prediction result")
            attr_pred.save_prediction(img_ids, prob_preds, cfg.output_path)

    print("save final prediction result")
    attr_pred.save_prediction(img_ids, prob_preds, cfg.output_path)

def _non_dist_test_cate_attr(model, dataset, cfg, validate=False):
    data_loader = build_dataloader(
        dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)

    print('dataloader built')

    model = MMDataParallel(model, device_ids=cfg.gpus.test).cuda()
    model.eval()

    attr_calculator = AttrCalculator(
        cfg,
        topns=[3, 5],
        show_attr_name=True,
        attr_name_file=cfg.data.test['attr_cloth_file'])
    cate_calculator = CateCalculator(cfg, topns=[1, 3, 5])

    for batch_idx, testdata in enumerate(data_loader):
        imgs = testdata['img']
        landmark = testdata['landmark']
        attr = testdata['attr']
        cate = testdata['cate']

        attr_pred, cate_pred = model(
            imgs, attr, landmark=landmark, return_loss=False)

        attr_calculator.collect_result(attr_pred, attr)
        cate_calculator.collect_result(cate_pred, cate)

        if batch_idx % cfg.print_interval == 0:
            attr_calculator.show_result(batch_idx)
            cate_calculator.show_result(batch_idx)

    attr_calculator.show_result()
    attr_calculator.show_per_attr_result()
    cate_calculator.show_result()


def _dist_test_attr(model, dataset, cfg, validate=False):
    raise NotImplementedError


def _dist_test_cate_attr(model, dataset, cfg, validate=False):
    raise NotImplementedError
