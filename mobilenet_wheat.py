from utils import *
from init import *
from CustomClasses import NeuralNetwork, ImageDataset, CNN
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
import torch
from torch.optim import lr_scheduler
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision import transforms

from wheat_data_utils import WheatImgDataset, get_class_weights, oversampler, save_csv
from model_utils import *
from torchinfo import summary
from wheat_data_prep import (
    TRAINING_DATA,
    VALIDATION_DATA,
    TESTING_DATA,
    TRAIN_SAMPLER,
    CLASSES,
    data_loader,
)

from datetime import datetime
import sys
import time
from tempfile import TemporaryDirectory


TRAIN_ACC = []
TRAIN_LOSS = []
VAL_ACC = []
VAL_LOSS = []
DURATIONS = []
TOTAL_TIME = 0

args = cmd_args()
EPOCHS = args.epochs
BATCH_SIZE = args.batch
LR = args.lr  # [backbone_lr, classifier_lr]
FREEZE = args.freeze
MODEL_PATH = args.load
IS_TRAIN = args.train
# IS_TEST = args.test
IS_EVAL = args.eval
IS_SCHEDUL = args.scheduler
W_DECAY = args.decay


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice: ", DEVICE)


IMAGE, _ = TRAINING_DATA[0]
C, H, W = image_shape(IMAGE)
print(f"Images shape: {C, H, W}")


print(f"\nClasses are:\n{CLASSES}\n")

dev = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_LOADER = data_loader(
    TRAINING_DATA, device=dev, batch_size=BATCH_SIZE, sampler=TRAIN_SAMPLER
)
TEST_LOADER = data_loader(
    TRAINING_DATA,
    device=dev,
    batch_size=BATCH_SIZE,
)
VAL_LOADER = data_loader(
    VALIDATION_DATA,
    device=dev,
    batch_size=BATCH_SIZE,
)

from torch.nn import Dropout, Dropout2d

MODEL = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).to(DEVICE)

# freeze head for feature extraction
if FREEZE:
    for param in MODEL.parameters():
        param.requires_grad = False


MODEL.classifier[2] = nn.Dropout(p=0.5, inplace=True)
MODEL.classifier[3] = nn.Linear(in_features=1024, out_features=len(CLASSES))
MODEL.classifier.insert(0, nn.Dropout(p=0.3, inplace=True))


# summary(MODEL, input_size=(1, 3, 32, 32), device="cpu", verbose=1)


def main():

    t0 = time.perf_counter()

    model = MODEL

    if MODEL_PATH:
        model = load_model(MODEL_PATH, model)

    model.to(DEVICE)
    # loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)  # for single class
    loss_fn = nn.CrossEntropyLoss()  # for single class
    # loss_fn = nn.BCEWithLogitsLoss()  # for multiple classes
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

    opt_algo = torch.optim.AdamW
    optimizer = opt_algo(model.classifier.parameters(), lr=LR[1], weight_decay=W_DECAY)
    # for unfrozen backbone
    if not FREEZE:
        optimizer = opt_algo(
            [
                {"params": model.features.parameters(), "lr": LR[0]},
                {"params": model.classifier.parameters(), "lr": LR[1]},
            ],
            weight_decay=W_DECAY,
        )
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

    script_name = sys.argv[0].strip(".py")
    TAG = f"{script_name}{'_frozen' if FREEZE else '_unfrozen'}"
    file_name = f"{TAG}_batch-size:{BATCH_SIZE}_lr:{LR}_w-decay:{W_DECAY}_{datetime.now().strftime("%H:%M:%S-%d.%m.%Y")}"

    if IS_TRAIN:
        for t in range(EPOCHS):
            epochs = []
            start_t = time.perf_counter()
            print(f"Epoch {t+1}\n-------------------------------")
            train_acc, train_loss = train(TRAIN_LOADER, model, loss_fn, optimizer)
            val_acc, val_loss = test(VAL_LOADER, model, loss_fn)
            end_t = time.perf_counter()
            DURATIONS.append(end_t - start_t)
            TRAIN_ACC.append(train_acc)
            TRAIN_LOSS.append(train_loss)
            VAL_ACC.append(val_acc)
            VAL_LOSS.append(val_acc)
            epochs.append(t)

            df = pd.DataFrame(
                {
                    "Epoch": epochs,
                    "Training_accuracy": TRAIN_ACC,
                    "Training_loss": TRAIN_LOSS,
                    "Validation_accuracy": TRAIN_ACC,
                    "Validation_loss": TRAIN_LOSS,
                    "Duration": DURATIONS,
                }
            )
            save_csv(path=f"output_data/{TAG}/{file_name}.csv", data=df)

            # scheduler.step()
            if IS_SCHEDUL:
                scheduler.step(val_loss)

        print("\nEnd of Training!\n")
        TOTAL_TIME = time.perf_counter() - t0
        print("###############################\n")
        print(
            f"# Training time: {time.strftime("%H:%M:%S", time.gmtime(TOTAL_TIME))} #\n"
        )
        print("###############################\n")

        save_model(model, path=f"models/{file_name}.pth")

        # df = pd.DataFrame(
        #     {
        #         "Epoch": range(1, EPOCHS + 1),
        #         "Accuracy": ACC,
        #         "Average_loss": AVG_LOSS,
        #         "Duration": DURATIONS,
        #     }
        # )
        # save_csv(path=f"output_data/{TAG}/{file_name}.csv", data=df)

    if IS_EVAL:
        print("Evaluation...")
        t1 = time.perf_counter()

        classes_list = list(CLASSES)
        eval_avg_acc(model, TESTING_DATA, print_res=True)
        eval_per_class(TEST_LOADER, model, classes_list)

        print("End of Evaluation!")
        eval_time = time.perf_counter() - t1
        TOTAL_TIME += eval_time
        print("\n##########################\n")
        print(
            f"# Evaluation time: {time.strftime("%H:%M:%S", time.gmtime(eval_time))} #\n"
        )
        print("##########################\n")
        print(
            f"# Total elapsed time: {time.strftime("%H:%M:%S", time.gmtime(TOTAL_TIME))} #\n"
        )
        print("##########################\n")


if __name__ == "__main__":
    main()
