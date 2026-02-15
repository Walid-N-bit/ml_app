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
from cifar_data_prep import TRAINING_DATA, TESTING_DATA, data_loader, CLASSES

from torchinfo import summary

from datetime import datetime
import sys, time
from tempfile import TemporaryDirectory


ACC = []
AVG_LOSS = []
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
TRAIN_LOADER = data_loader(TRAINING_DATA, device=dev, batch_size=BATCH_SIZE)
TEST_LOADER = data_loader(TESTING_DATA, device=dev, batch_size=BATCH_SIZE)


MODEL = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).to(DEVICE)

# freeze head for feature extraction
if FREEZE:
    for param in MODEL.parameters():
        param.requires_grad = False


MODEL.classifier[3] = nn.Linear(in_features=1024, out_features=len(CLASSES))


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
            start_t = time.perf_counter()
            print(f"Epoch {t+1}\n-------------------------------")
            val_loss = train(TRAIN_LOADER, model, loss_fn, optimizer)
            acc, loss = test(TEST_LOADER, model, loss_fn)
            elapsed_t = time.perf_counter() - start_t
            DURATIONS.append(elapsed_t)
            ACC.append(acc)
            AVG_LOSS.append(loss)

            df = pd.DataFrame(
                {
                    "Epoch": range(1, EPOCHS + 1),
                    "Accuracy": ACC,
                    "Average_loss": AVG_LOSS,
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
        eval_general(model, TESTING_DATA, DEVICE, classes_list)
        eval_per_class(TEST_LOADER, model, classes_list)

        print("End of Evaluation!")
        eval_time = time.perf_counter() - t1
        TOTAL_TIME += eval_time
        print("##########################\n")
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
