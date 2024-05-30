import os
import torch
import data_setup, engine, model_builder, utils
import argparse
import pandas as pd
from pathlib import Path

from torchvision import transforms

from timeit import default_timer as timer

def main():

    # default values of Hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    parser = argparse.ArgumentParser(description='To set hyperparameters for the model')

    parser.add_argument('-lr','--learning_rate', type=float, default=LEARNING_RATE, help='learning rate hyperparameter for the optimizer')
    parser.add_argument('-bz','--batch_size', type=int, default=BATCH_SIZE, help='batch size for model training')
    parser.add_argument('-hdu','--hidden_units', type=int, default=HIDDEN_UNITS, help='hidden units hyperparameter for the model')
    parser.add_argument('-eps','--num_epochs', type=int, default=NUM_EPOCHS, help='number of training epochs for the model')

    args = parser.parse_args()

    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate

    # setup directories
    train_dir = "data/cifar10_images/train"
    test_dir = "data/cifar10_images/test"

    # setup target device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])

    # create dataloader with data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(
        train_dir = train_dir,
        test_dir = test_dir,
        train_transform = transform,
        test_transform = transform,
        batch_size = BATCH_SIZE
    )

    # create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # set up loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("model training initiated")
    
    start_time = timer()
    model_results = engine.train(
                                model=model,
                                train_dataloader=train_dataloader,
                                test_dataloader=test_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                epochs=NUM_EPOCHS,
                                device=device 
    )

    end_time = timer()
    total_time = end_time - start_time
    
    model_file_name=f"05_TinyVGG_lr_{LEARNING_RATE}_bz_{BATCH_SIZE}_hdu_{HIDDEN_UNITS}_eps_{NUM_EPOCHS}___time_{total_time:.2f}"

    # Save the model with utils.py
    utils.save_model(
        model=model,
        target_dir="models",
        model_name=f"{model_file_name}.pth"
    )

    # save results as .csv
    model_results = pd.DataFrame(model_results)
    result_dir_path = Path("results")
    result_save_path = result_dir_path / f"{model_file_name}.csv"

    print(f"[INFO] Saving results to: {result_save_path}")
    model_results.to_csv(result_save_path)

if __name__ == '__main__':
    main()