from .aug_tta import AugTTA
from .class_tta import ClassTTA
from .gps import GPS
from .naive import MeanTTA, MaxTTA
from .agg_tta import AggTTA


def get_agg_model(
    agg_name: str, 
    budget: int | None = None,
    num_classes: int | None = None,
    n_subpolicies: int | None = 3,
):
    match agg_name:
        case 'AugTTA':
            return AugTTA(n_augs=budget)
        case 'ClassTTA':
            return ClassTTA(n_augs=budget, num_classes=num_classes)
        case 'MeanTTA':
            return MeanTTA()
        case 'MaxTTA':
            return MaxTTA()
        case 'GPS':
            return GPS(n_subpolicies=n_subpolicies)
        case _:
            raise ValueError(agg_name)