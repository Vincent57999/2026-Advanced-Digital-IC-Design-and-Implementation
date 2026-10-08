import argparse
import importlib.util
import inspect
import os
import sys

import numpy as np
import torch
import torch.nn as nn

N_CLASSES = 8
MAX_WEIGHT_MB = 2.0
LINE = "=" * 46


def check(args):
    for path in (args.model, args.weights, args.data):
        if not os.path.isfile(path):
            raise RuntimeError(f"File not found: {path}")

    size_mb = os.path.getsize(args.weights) / 2 ** 20
    over = size_mb > MAX_WEIGHT_MB
    print(f"  Weight file: {size_mb:.2f} MB" +
          (f"  -- OVER THE {MAX_WEIGHT_MB:.0f} MB LIMIT" if over else "  OK"))

    d = np.load(args.data)
    x_test = d["test_images"]
    y_test = d["test_labels"].reshape(-1).astype(np.int64)

    spec = importlib.util.spec_from_file_location("model", args.model)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model"] = mod
    spec.loader.exec_module(mod)

    mean, std = getattr(mod, "MEAN", None), getattr(mod, "STD", None)
    if mean is None or std is None:
        # Not declared: assume the model was trained on [0, 1] pixels.
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        print("  MEAN/STD not declared, assuming inputs in [0, 1].")
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)

    cls = getattr(mod, "Model", None)
    if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
        found = [n for n, o in inspect.getmembers(mod, inspect.isclass)
                 if issubclass(o, nn.Module) and o.__module__ == mod.__name__]
        raise RuntimeError(
            "model.py must define a class named 'Model' that subclasses nn.Module. "
            + (f"Found instead: {', '.join(found)}." if found else
               "No nn.Module subclass was found."))
    model = cls()
    model.eval()

    try:
        sd = torch.load(args.weights, map_location="cpu", weights_only=True)
    except Exception:
        sd = None
    if not isinstance(sd, dict):
        raise RuntimeError("weight.pth could not be read as a state_dict. Save the "
                           "parameters only: torch.save(model.state_dict(), 'weight.pth'), "
                           "not torch.save(model).")
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        raise RuntimeError("weight.pth does not match the Model in model.py -- the layer "
                           f"names or shapes differ.\n{e}")

    x = torch.from_numpy(x_test).float().permute(0, 3, 1, 2)
    x = (x / 255.0 - mean_t) / std_t
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            out = model(x[i:i + 256])
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.shape[1] != N_CLASSES:
                raise RuntimeError(f"Model output has {out.shape[1]} columns, "
                                   f"expected {N_CLASSES} (one score per class).")
            if not torch.isfinite(out).all():
                raise RuntimeError("Model output contains NaN or Inf.")
            preds.append(out.argmax(1))
    return float((torch.cat(preds).numpy() == y_test).mean())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Check that your BloodMNIST submission runs, and report its test accuracy.",
        epilog="Put model.py, weight.pth and bloodmnist.npz in the same folder as this "
               "script, then run: python test.py")
    ap.add_argument("--model", default=os.path.join(here, "model.py"))
    ap.add_argument("--weights", default=os.path.join(here, "weight.pth"))
    ap.add_argument("--data", default=os.path.join(here, "bloodmnist.npz"))
    args = ap.parse_args()

    print(LINE)
    print("  BloodMNIST Submission Check")
    print(LINE)
    try:
        acc = check(args)
    except Exception as e:
        print(f"  Error: {e}" if str(e) else "  Error")
        print(LINE)
        sys.exit(1)
    print("  Model runs correctly.")
    print(f"  Test accuracy: {acc * 100:.2f}%")
    print(LINE)


if __name__ == "__main__":
    main()
