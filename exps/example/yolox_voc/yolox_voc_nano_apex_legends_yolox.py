#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # クラス数の設定
        self.num_classes = 3

        # YOLOX_NANOのアーキテクチャ設定（Nano: depth=0.33, width=0.25）
        self.depth = 0.33
        self.width = 0.25

        # 高精度を狙うため、入力解像度を上げる（例：640×640）
        self.input_size = (416, 416)
        self.test_size = (416, 416)

        # 学習エポック数を増加して十分な収束を図る（例：500エポック）
        self.max_epoch = 300

        # 最終フェーズではAugmentationなしで微調整するための期間を延長（例：50エポック）
        self.no_aug_epochs = 50

        # ウォームアップエポック（例：10エポック）
        self.warmup_epochs = 10

        # バッチサイズあたりの学習率設定（学習率はバッチサイズに依存するため調整が必要）
        self.basic_lr_per_img = 0.01 / 64

        # Mosaicのスケール（学習時のスケール変動）
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)

        # データ拡張：反転やHSV変換の確率（適度な拡張で過剰な変形を防ぐ）
        self.flip_prob = 0.5
        self.hsv_prob = 0.5

        # mixupはオフ（検証時の挙動を安定させるため）
        self.enable_mixup = False

        # 評価時の信頼度閾値・NMS閾値の調整
        # ※高い適合率・再現率を得るため、低めのconfで候補を多めに残す調整例
        self.test_conf = 0.05
        self.nmsthre = 0.65

        # expファイル名（自動取得）
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import VOCDetection, TrainTransform

        return VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import VOCDetection, ValTransform
        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            image_sets=[('2007', 'test')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
