from settings import config


def load_model(model_name, model_path):
    """
    """
    from .model import MTTS_CAN
    model_args = config.model.args
    model = MTTS_CAN(
        n_frame=model_args.frame_depth,
        nb_filters1=model_args.nb_filters1,
        nb_filters2=model_args.nb_filters2,
        input_shape=model_args.input_shape,
    )
    model.load_weights(model_path)
    return model
