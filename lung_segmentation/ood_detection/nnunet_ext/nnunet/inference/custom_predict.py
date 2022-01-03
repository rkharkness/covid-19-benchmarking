import torch
# Dataloader
# Model loader
# Feature extraction
def save_features(self, data: np.ndarray, do_mirroring: bool = True,
            mirror_axes: Tuple[int] = None,
            use_sliding_window: bool = True, step_size: float = 0.5,
            use_gaussian: bool = True, pad_border_mode: str = 'constant',
            pad_kwargs: dict = None, all_in_gpu: bool = False,
            verbose: bool = True, mixed_precision=True, tta: int = -1, mcdo: int = -1, 
            features_dir=None, feature_paths=None) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Basically a copy of predict_preprocessed_data_return_seg_and_softmax, but stores features instead of making
        predictions.
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().save_features(data,
            do_mirroring=do_mirroring,
            mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window,
            step_size=step_size, use_gaussian=use_gaussian,
            pad_border_mode=pad_border_mode,
            pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
            verbose=verbose,
            mixed_precision=mixed_precision, tta=tta, mcdo=mcdo, 
            features_dir=features_dir, feature_paths=feature_paths)
        self.network.do_ds = ds
        return ret

def predict_cases(model, fold_no, checkpoint_path, output_probabilities: bool = False, tta: int = -1, mcdo: int = -1, no_softmax=False,
                  features_dirs=None, feature_paths=None):
    print("emptying cuda cache")
    torch.cuda.empty_cache()
    print("loading parameters for folds,", folds)    
    model = model.load_state_dict(torch.load(f"{checkpoint_path}_{fold_no}.pth"))

    # assumed from existing code
    force_separate_z = None
    interpolation_order = 1
    interpolation_order_z = 0


device = torch.device('cuda')
model = VGGNestedUNet(num_classes=1)
model = model.to(device)
