from src.utils import get_device
from src.dataset import get_datasets
from src.utils import get_dataloaders_for, seed_everything, collate_fn
from src.io import load_model_weights, save_model_weights, save_results
# from src.models.fpsnet import FPSNet
# from src.models.retinanet import RetinaNet
from src.models.retinanet_cbam import RetinaNetCBAM
from src.models.retinanet import RetinaNet
from src.models.fasterrcnn import FasterRCNN
from src.models.fasterrcnn_cbam import FasterRCNNCBAM
import os, sys

def main():
    DEVICE = get_device()
    N_ATT = 35 # this is an upper bound on the number of craters
    BATCH_SIZE = 12
    EPOCHS = 20
    DIM=200
    SEED=101
    os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), ".cache", "torch")

    seed_everything(SEED)

    if len(sys.argv) < 2:
        print("Usage: python main.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    train_dataset, test_dataset = get_datasets(dataset_path=dataset_path, dim=(DIM, DIM))
    train_dataloader, test_dataloader = get_dataloaders_for(train_dataset, test_dataset, batch_size=BATCH_SIZE, seed=SEED, collate_fn=collate_fn)
    test_dataset.png_files = test_dataset.png_files[:250]
    print("Train Dataset:", train_dataset)
    print("Test Dataset:", test_dataset)

    # model = RetinaNet(device=DEVICE)
    # model.load_state_dict(load_model_weights("saved_weights", "crater_retinanet_weights.pth"))
    # model.train_model(train_dataloader, 0.01, EPOCHS)
    # test_metrics = model.test_model(test_dataloader)
    # save_results(test_metrics, "retinanet.json")
    # save_model_weights(model, "saved_weights", "crater_retinanet_weights.pth")

    # model = RetinaNetCBAM(device=DEVICE)
    # model.load_state_dict(load_model_weights("saved_weights", "crater_retinanet_cbam_weights.pth"))
    # model.train_model(train_dataloader, 0.01, EPOCHS)
    # test_metrics = model.test_model(test_dataloader)
    # save_results(test_metrics, "retinanet_cbam.json")
    # save_model_weights(model, "saved_weights", "crater_retinanet_cbam_weights.pth")

    model = FasterRCNN(device=DEVICE)
    model.load_state_dict(load_model_weights("saved_weights", "crater_faster_rcnn_weights.pth"))
    # model.train_model(train_dataloader, 0.01, EPOCHS)
    test_metrics = model.test_model(test_dataloader)
    save_results(test_metrics, "faster_rcnn.json")
    # save_model_weights(model, "saved_weights", "crater_faster_rcnn_weights.pth")

    model = FasterRCNNCBAM(device=DEVICE)
    model.load_state_dict(load_model_weights("saved_weights", "crater_faster_rcnn_cbam_weights.pth"))
    # model.train_model(train_dataloader, 0.01, EPOCHS)
    test_metrics = model.test_model(test_dataloader)
    # result_lines = []
    # # print("-- Test Metrics --")
    # # for k, v in test_metrics.items():
    # #     print(f"\"{k}\": {v},")
    # #     result_lines.append(f"\"{k}\": {v}")
    save_results(test_metrics, "faster_rcnn_cbam.json")
    # save_model_weights(model, "saved_weights", "crater_faster_rcnn_cbam_weights.pth")

if __name__ == "__main__":
    main()
