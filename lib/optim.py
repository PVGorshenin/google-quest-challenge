import torch.optim as optim


def configure_optimizers(model, lr):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay_rate": 0.01
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay_rate": 0.0
        },
    ]
    optimizer = optim.Adam(
        optimizer_grouped_parameters,
        lr=lr,
    )
    return optimizer


def get_model_optimizer_exp(model, lr):
    params = list(model.named_parameters())

    def is_backbone(n):
        return "bert" in n

    optimizer_grouped_parameters = [
        {"params": [p for n, p in params if is_backbone(n)], "lr": lr},
        {"params": [p for n, p in params if not is_backbone(n)], "lr": lr * 500},
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        weight_decay=0
    )

    return optimizer