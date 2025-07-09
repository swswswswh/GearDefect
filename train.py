from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolo11s-cls.pt")  # load a pretrained model (recommended for training)

    model.train(
        data="data_set",  # path to data.yaml file
        epochs=100,  # number of training epochs
        batch=0.9,  # batch size
        workers=4,
        device=0,  # device to train on (0 for GPU, 'cpu' for CPU)
        lr0=0.001,
        warmup_epochs=3,
        weight_decay=0.01,
        hsv_h=0.015,  # 图像HSV-Hue增强
        hsv_s=0.7,    # 图像HSV-Saturation增强
        hsv_v=0.4,    # 图像HSV-Value增强
        flipud=0.5,   # 上下翻转概率
        fliplr=0.5,   # 左右翻转概率
        mosaic=1.0,   # 使用mosaic数据增强
        mixup=0.2,    # 使用mixup数据增强
        copy_paste=0.3 # 使用copy-paste数据增强
    )