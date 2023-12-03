from model.yolo import Yolo_V3

import time
import torch

def main(nimble=False):
    
    model = Yolo_V3([[5,5,5],[5,5,5],[5,5,5]],5)
    start_time = time.time()
    
    if nimble:
        model = torch.cuda.nimble(model)
        model.prepare(torch.randr(1,3,416,416))
        prepare_time = time.time() - start_time
        print(f"nimble preparation time: {prepare_time}")
    
    start_time = time.time()
    model(torch.randr(1,3,416,46))
    train_time = time.time() - start_time

    if nimble:
        print(f"nimble train time: {train_time}")
    else:
        print(f"model train time: {train_time}")


if __name__ == '__main__':
    main()
