from typing import Dict, Any, Union, Optional
import matplotlib
import numpy

# TODO
Experiment = Any
ActiveRun = Any
Run = Any
Dataset = Any
plotly = Any
PIL = Any
pandas = Any

def set_experiment(
    experiment_name: Optional[str] = None, experiment_id: Optional[str] = None
) -> Experiment: ...
def start_run(
    run_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    run_name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
) -> ActiveRun: ...
def end_run(status: str = "FINISHED") -> None: ...
def active_run() -> Optional[ActiveRun]: ...
def last_active_run() -> Optional[Run]: ...
def get_run(run_id: str) -> Run: ...
def get_parent_run(run_id: str) -> Optional[Run]: ...
def log_param(key: str, value: Any) -> Any: ...
def set_experiment_tag(key: str, value: Any) -> None: ...
def set_tag(key: str, value: Any) -> None: ...
def delete_tag(key: str) -> None: ...
def log_metric(key: str, value: float, step: Optional[int] = None) -> None: ...
def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None: ...
def log_params(params: Dict[str, Any]) -> None: ...
def log_input(
    dataset: Dataset,
    context: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> None: ...
def set_experiment_tags(tags: Dict[str, Any]) -> None: ...
def set_tags(tags: Dict[str, Any]) -> None: ...
def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None: ...
def log_artifacts(local_dir: str, artifact_path: Optional[str] = None) -> None: ...
def log_text(text: str, artifact_file: str) -> None: ...
def log_dict(dictionary: Dict[str, Any], artifact_file: str) -> None: ...
def log_figure(
    # figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"],
    figure: Any,
    artifact_file: str,
    *,
    save_kwargs: Optional[Dict[str, Any]] = None,
) -> None: ...
def log_image(
    # image: Union["numpy.ndarray", "PIL.Image.Image"],
    image: Any,
    artifact_file: str,
) -> None: ...
def log_table(
    # data: Union[Dict[str, Any], "pandas.DataFrame"],
    data: Dict[str, Any],
    artifact_file: str,
) -> None: ...
def load_table(
    artifact_file: str,
    run_ids: Optional[list[str]] = None,
    extra_columns: Optional[list[str]] = None,
) -> Any: ...  # Returns "pandas.DataFrame"
def get_experiment(experiment_id: str) -> Experiment: ...
def get_experiment_by_name(name: str) -> Optional[Experiment]: ...
def search_experiments(
    view_type: int = 1,  # ViewType.ACTIVE_ONLY
    max_results: Optional[int] = None,
    filter_string: Optional[str] = None,
    order_by: Optional[list[str]] = None,
) -> list[Experiment]: ...
def create_experiment(
    name: str,
    artifact_location: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> str: ...
def delete_experiment(experiment_id: str) -> None: ...
def delete_run(run_id: str) -> None: ...
def get_artifact_uri(artifact_path: Optional[str] = None) -> str: ...
def search_runs(
    experiment_ids: Optional[list[str]] = None,
    filter_string: str = "",
    run_view_type: int = 1,  # ViewType.ACTIVE_ONLY,
    max_results: int = 100000,
    order_by: Optional[list[str]] = None,
    output_format: str = "pandas",
    search_all_experiments: bool = False,
    experiment_names: Optional[list[str]] = None,
) -> list[Run]: ...  # Union[list[Run], "pandas.DataFrame"]
def autolog(
    log_input_examples: bool = False,
    log_model_signatures: bool = True,
    log_models: bool = True,
    log_datasets: bool = True,
    disable: bool = False,
    exclusive: bool = False,
    disable_for_unsupported_versions: bool = False,
    silent: bool = False,
    extra_tags: Optional[Dict[str, str]] = None,
) -> None: ...
