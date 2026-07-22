import utils.utils_net as networks


def build_models(device, include_alignment=True):
    dem_model = networks.PIDNet().to(device)
    task_model = networks.SfPNet().to(device)
    fa_model = networks.FeatureAlignment().to(device) if include_alignment else None
    return dem_model, task_model, fa_model
