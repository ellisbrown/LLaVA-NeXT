from llava.train.train import train

if __name__ == "__main__":
    import warnings

    # Ignore warnings that are shown multiple times
    warnings.simplefilter("once")
    warnings.warn("Only showing each warning once")

    train()
