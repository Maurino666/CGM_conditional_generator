import torch

from data_gathering_pipeline import gather_data
from src.models import *


def main():

    train_loader, val_loader = gather_data(
        ["glucose"],
        228,
        12,
        val_ratio = 0.2,
        random_state = 1,
        batch_size = 64,
        normalize=["glucose"],
    )

    _, input_dim = train_loader.dataset[0].shape

    hidden_dim = 64
    latent_dim = 16
    num_layers = 1

    model = RnnVaeModule(
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        latent_dim = latent_dim,
        num_layers = num_layers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = train_module(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        num_epochs = 3,
        device = device,
    )

    print("History: ", history)

    print("Starting TimeGan Training")
    model2 = TimeGanModule(
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        g_steps_per_iter=1
    )

    model2.set_phase("ae")

    history = train_module(
        model = model2,
        train_loader = train_loader,
        val_loader = val_loader,
        num_epochs=3,
        device = device,
    )

    print("History ae: ", history)


    model2.set_phase("sup")

    history = train_module(
        model = model2,
        train_loader = train_loader,
        val_loader = val_loader,
        num_epochs=3,
        device = device,
    )

    print("History sup: ", history)


    model2.set_phase("adv")

    history = train_module(
        model = model2,
        train_loader = train_loader,
        val_loader = val_loader,
        num_epochs=3,
        device = device,
    )

    print("History vae: ", history)

if __name__ == "__main__":
    main()