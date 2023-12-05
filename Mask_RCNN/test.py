import torchvision
import torch
import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main(nimble):
    device = torch.device("cuda:0")

    model_input = torch.rand([1, 3, 224, 224]).to(device)
    prepare_input = torch.rand([1, 3, 224, 224]).to(device)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    model.to(device)
    model.eval()

    start_time = time.time()
    if nimble:
        model = torch.cuda.Nimble(model)
        model.prepare(prepare_input, training=False)
        init_end = time.time()
        init_time = init_end - start_time
        logger.info(f"nimble init time: {init_time}")
    
    train_start = time.time()
    model(model_input)
    train_time = time.time() - init_end
    
    if nimble:
        logger.info(f"nimble train time: {train_time}")
    else:
        logger.info(f"model train time: {train_time}")


if __name__ == '__main__':
    main(nimble=True)