import timm 
from timm.data import resolve_data_config

def get_timm_pretrained(
    model_name: str, 
    num_classes: int = 1000
) -> tuple:
    model = timm.create_model(
        model_name, pretrained=True, num_classes=num_classes)
    member_fns = {}
    return model, member_fns

def print_timm_model_info(model_name: str) -> None:
    model = timm.create_model(model_name)
    data_cfg = resolve_data_config(model=model_name)
    print(model)
    print(data_cfg)

if __name__ == '__main__':
    import sys
    model_name = sys.argv[1]
    print_timm_model_info(model_name)