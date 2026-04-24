import inspect
import sys
from dataclasses import dataclass
from typing import Union

from pytorch_lightning.callbacks.progress.rich_progress import *
from pytorch_lightning.utilities import rank_zero_only
from rich import print, reconfigure
from rich.console import RenderableType
from rich.progress import ProgressColumn
from rich.style import Style
from rich.text import Text


@rank_zero_only
def print_only(message: str):
    print(message)


@dataclass
class RichProgressBarTheme:
    """Styles to associate to different base components."""

    description: Union[str, Style] = "#FF4500"
    progress_bar: Union[str, Style] = "#f92672"
    progress_bar_finished: Union[str, Style] = "#b7cc8a"
    progress_bar_pulse: Union[str, Style] = "#f92672"
    batch_progress: Union[str, Style] = "#fc608a"
    time: Union[str, Style] = "#45ada2"
    processing_speed: Union[str, Style] = "#DC143C"
    metrics: Union[str, Style] = "#228B22"


class BatchesProcessedColumn(ProgressColumn):
    def __init__(self, style: Union[str, Style]):
        self.style = style
        super().__init__()

    def render(self, task) -> RenderableType:
        total = task.total if task.total != float("inf") else "--"
        return Text(f"{int(task.completed)}/{int(total)}", style=self.style)


class MyMetricsTextColumn(ProgressColumn):
    """A column containing text."""

    def __init__(self, style):
        self._tasks = {}
        self._current_task_id = 0
        self._metrics = {}
        self._style = style
        super().__init__()

    def update(self, metrics):
        self._metrics = metrics

    def render(self, task) -> Text:
        text = ""
        for k, v in self._metrics.items():
            text += f"{k}: {round(v, 3) if isinstance(v, float) else v} "
        return Text(text, justify="left", style=self._style)


class MyRichProgressBar(RichProgressBar):
    """Use rich progress interactively and epoch summaries in captured logs."""

    def __init__(self, *args, **kwargs):
        force_single_line = kwargs.pop("force_single_line", False)
        super().__init__(*args, **kwargs)
        self._single_line_epoch_mode = force_single_line or self._should_use_single_line_epoch_mode()
        self._last_summarized_epoch = None

    def _should_use_single_line_epoch_mode(self) -> bool:
        stdout = getattr(sys, "stdout", None)
        is_tty = getattr(stdout, "isatty", lambda: False)
        return not bool(is_tty())

    @staticmethod
    def _format_metric_value(value):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def _build_epoch_summary(self, trainer, pl_module) -> str:
        metrics = self.get_metrics(trainer, pl_module)
        preferred_order = (
            "train/loss",
            "train/learning_rate",
            "val/loss",
            "val/si_snr",
        )
        ignored_metrics = {"v_num"}
        metric_parts = []
        used_metrics = set()

        for name in preferred_order:
            if name in metrics:
                metric_parts.append(f"{name}: {self._format_metric_value(metrics[name])}")
                used_metrics.add(name)

        for name, value in metrics.items():
            if name in ignored_metrics or name in used_metrics:
                continue
            metric_parts.append(f"{name}: {self._format_metric_value(value)}")

        if trainer.max_epochs is not None:
            epoch_text = f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
        else:
            epoch_text = f"Epoch {trainer.current_epoch + 1}"

        if metric_parts:
            return f"{epoch_text} {' '.join(metric_parts)}"
        return epoch_text

    def _print_epoch_summary_once(self, trainer, pl_module) -> None:
        if self._last_summarized_epoch == trainer.current_epoch:
            return
        print_only(self._build_epoch_summary(trainer, pl_module))
        self._last_summarized_epoch = trainer.current_epoch

    def _init_progress(self, trainer):
        if self._single_line_epoch_mode:
            return
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            if hasattr(self._console, "_live_stack"):
                if len(self._console._live_stack) > 0:
                    self._console.clear_live()
            else:
                self._console.clear_live()
            metric_sig = inspect.signature(MetricsTextColumn.__init__)
            if "text_delimiter" in metric_sig.parameters and "metrics_format" in metric_sig.parameters:
                self._metric_component = MetricsTextColumn(
                    trainer,
                    self.theme.metrics,
                    " ",
                    ".3f",
                )
            else:
                self._metric_component = MetricsTextColumn(trainer, self.theme.metrics)
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            self._progress_stopped = False

    def on_validation_end(self, trainer, pl_module) -> None:
        if self._single_line_epoch_mode:
            if trainer.state.fn == "fit" and not trainer.sanity_checking:
                self._print_epoch_summary_once(trainer, pl_module)
            self.reset_dataloader_idx_tracker()
            return
        super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._single_line_epoch_mode:
            num_val_batches = getattr(trainer, "num_val_batches", 0)
            has_validation = any(num_val_batches) if isinstance(num_val_batches, list) else bool(num_val_batches)
            if not has_validation:
                self._print_epoch_summary_once(trainer, pl_module)
            return
        super().on_train_epoch_end(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if self._single_line_epoch_mode:
            return
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._single_line_epoch_mode:
            return
        super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._single_line_epoch_mode:
            return
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def _get_train_description(self, current_epoch: int) -> str:
        max_e = self.trainer.max_epochs
        if max_e is not None:
            display = min(current_epoch + 1, max_e)
            train_description = f"Epoch {display}/{max_e}"
        else:
            train_description = f"Epoch {current_epoch + 1}"
        if len(self.validation_description) > len(train_description):
            train_description = f"{train_description:{len(self.validation_description)}}"
        return train_description
