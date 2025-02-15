# Training Capabilities

The training functionality is provided by the [`train`](cli.py) command. This command starts the YOLO model training pipeline with configurable parameters and automatic logging.

## Command Usage

```sh
python3 cli.py train --mode <bricks|studs> [OPTIONS]
```

### Options

- --mode: (required) Specifies the training mode. Accepted values: bricks or studs.
- --epochs: Sets the number of training epochs. (Default: 20)
- --batch-size: Sets the batch size for training. (Default: 16)
- --use-pretrained: Use a pre-trained LEGO model instead of training from scratch.
- --cleanup / --no-cleanup: Enable or disable removal of temporary directories (cache, logs, results) after training. (Default: cleanup enabled)
- --force-extract: Force re-extraction of the dataset even if it already exists.
- --show-results / --no-show-results: Enable or disable the display of training session results after execution. (Default: show-results enabled)

### Example

Start a training session in "bricks" mode using a pre-trained model, with 10 epochs and a batch size of 16:

```sh
python3 cli.py train --mode bricks --use-pretrained --epochs 10 --batch-size 16
```

### How It Works

The command assembles the training arguments and invokes train.py with the specified options.
Logging is configured via the EmojiFormatter and setup_logging functions, ensuring that all training steps are logged with visual cues.
Post-training, if enabled, temporary directories (such as cache and results) are cleaned up automatically.

For more details on the training process and logging configuration, please refer to the train.py and cli.py files.