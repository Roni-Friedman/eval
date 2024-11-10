"""
Unitxt - Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI
https://github.com/IBM/unitxt
https://arxiv.org/abs/2401.14019
"""

# Standard
from dataclasses import dataclass, field
from enum import Enum, auto
from uuid import uuid4
import json
import os
import shutil

# Third Party
from lm_eval.tasks.unitxt import task
import pandas as pd
import yaml

# First Party
from instructlab.eval.mmlu import MMLUBranchEvaluator

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)

TEMP_DIR_PREFIX = "unitxt_temp"
GT_OUTPUT = "gt_output"
INPUT = "input"
CONTEXT = "context"
INSTRUCTION = "instruction"


@dataclass
class RenameStep:
    origin_field: str
    target_field: str

    def to_dict(self):
        return {
            "__type__": "rename",
            "field_to_field": {self.origin_field: self.target_field},
        }


class TaskType(Enum):
    QA = auto()
    SUMMARIZATION = auto()
    GENERATION = auto()
    RAG = auto()

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls[name.upper()]
        except KeyError:
            valid_values = ", ".join([task.name.lower() for task in cls])
            raise ValueError(
                f"'{name}' is not a valid TaskType. Available values are: {valid_values}"
            )

    def get_required_fields(self):
        fields = [INSTRUCTION, INPUT]
        match TaskType:
            case TaskType.RAG:
                fields.append(CONTEXT)
            case _:
                pass
        return fields


@dataclass
class UserUnitxtCardData:
    task_type: TaskType
    use_llmaaj: bool
    file_columns: list
    input_fields: dict = field(init=False)
    metric: str = field(init=False)
    templates: list = field(init=False)
    num_demos: int = field(init=False)
    pre_process_steps: list = field(init=False)

    def __post_init__(self):
        self.num_demos = 0
        self.metric = self._set_metric()
        self.input_fields, self.pre_process_steps = self._set_fields()
        self.templates = self._set_templates()

    def _set_metric(self) -> str:
        metric = "undefined"
        match self.task_type:
            case TaskType.QA:
                # TODO chage to generic metric
                metric = "metrics.llm_as_judge.rating.mistral_7b_instruct_v0_2_huggingface_template_mt_bench_single_turn"
                # TODO return when metric is available:
                # if GT_OUTPUT in self.file_columns:
                #     metric = f"{metric}_with_reference"
            case _:  # TODO support all other task types [take use llmaaj into account]
                pass
        return metric

    def _set_fields(self):
        available_columns = self.file_columns
        input_fields = {INSTRUCTION: "str"}
        pre_process_steps = []
        match self.task_type:
            case TaskType.QA:
                if "gt_output" in available_columns:
                    pre_process_steps.append(RenameStep("gt_output", "answer"))
                    input_fields["answer"] = "str"
                pre_process_steps.append(RenameStep("input", "question"))
                input_fields["question"] = "str"
            case _:
                pass
        return input_fields, pre_process_steps

    def _set_templates(self):
        templates = []
        match self.task_type:
            case TaskType.QA:
                # TODO need a template that uses the instruction field
                templates.append("templates.qa.open.empty")
            case _:
                pass
        return templates


class UnitxtEvaluator(MMLUBranchEvaluator):
    """
    An evaluator class, running Unitxt evaluation

    Attributes:
        model_path      Absolute path to or name of a huggingface model
        unitxt_recipe   unitxt recipe (see unitxt.ai for more information)
                        A Recipe holds a complete specification of a unitxt pipeline
                        Example: card=cards.wnli,template=templates.classification.multi_class.relation.default,max_train_instances=5,loader_limit=20,num_demos=3,demos_pool_size=10

    """

    name = "unitxt"

    def __init__(self, model_path, unitxt_recipe: str, output_dir: str = "eval_output"):
        unitxt_task, tasks_dir = self._assign_task_name_and_dir(output_dir)
        super().__init__(
            model_path=model_path, tasks_dir=tasks_dir, tasks=[unitxt_task], few_shots=0
        )
        self.unitxt_recipe = unitxt_recipe

    def _assign_task_name_and_dir(self, output_dir) -> tuple:
        task_name = str(uuid4())
        task_dir = os.path.join(output_dir, f"{TEMP_DIR_PREFIX}_{task_name}")
        return task_name, task_dir

    def _prepare_unitxt_files(self) -> None:
        taskname = self.tasks[0]
        yaml_file = os.path.join(str(self.tasks_dir), f"{taskname}.yaml")
        create_unitxt_pointer(self.tasks_dir)
        create_unitxt_yaml(
            yaml_file=yaml_file, unitxt_recipe=self.unitxt_recipe, task_name=taskname
        )

    def _remove_temp_unitxt_files(self, dir=None):
        if dir is None:
            dir = self.tasks_dir
        if os.path.basename(dir).startswith(
            TEMP_DIR_PREFIX
        ):  # to avoid unintended deletion if this class is inherited
            shutil.rmtree(dir)
        else:
            logger.warning(
                "unitxt tasks dir '%s' did not start with '%s' prefix and therefore was not deleted",
                os.path.basename(dir),
                TEMP_DIR_PREFIX,
            )

    def run(self, server_url: str | None = None) -> tuple:
        """
        Runs evaluation

        Attributes:
            server_url(str|None)    Model server endpoint (Ex: http://localhost:8000/v1) for the model being evaluated

        Returns:
            overall_scores      Average scores for the task group
            individual_scores   Individual scores for each task in the task group
        """
        self._prepare_unitxt_files()
        logger.debug(locals())
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        try:
            results = self._run_mmlu(server_url=server_url)
            taskname = self.tasks[0]
            global_scores = results["results"][taskname]
            global_scores.pop("alias")
            try:
                instances = results["samples"][taskname]
                instance_scores = {}
                metrics = [
                    metric.replace("metrics.", "")
                    for metric in instances[0]["doc"]["metrics"]
                ]
                for i, instance in enumerate(instances):
                    scores = {}
                    for metric in metrics:
                        scores[metric] = instance[metric][0]
                    instance_scores[i] = scores
            except KeyError as e:
                logger.error("Error in extracting single instance scores")
                logger.error(e)
                logger.error(e.__traceback__)
                instance_scores = None
        finally:
            self._remove_temp_unitxt_files()
        return global_scores, instance_scores


def create_unitxt_yaml(yaml_file: str, unitxt_recipe: str, task_name: str) -> None:
    data = {"task": task_name, "include": "unitxt", "recipe": unitxt_recipe}
    with open(yaml_file, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False)
    logger.debug("task %s unitxt recipe written to %s", task_name, yaml_file)


def create_unitxt_pointer(tasks_dir):
    class_line = "class: !function " + task.__file__.replace("task.py", "task.Unitxt")
    output_file = os.path.join(tasks_dir, "unitxt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(class_line)
    logger.debug("Unitxt task pointer written to %s", output_file)


class UnitxtFileEvaluator(UnitxtEvaluator):
    """
    An evaluator class, running evaluation on a given csv file, through Unitxt

    Attributes:
        model_path      Absolute path to or name of a huggingface model
        file_path       Path of a csv file
                        file must contain the following colums: 'input', 'instruction'
                        file may contain the following columns: 'gt_output', 'context'
        task_type       #TODO add description
        use_llmaaj      Optional bool, False by default
                        #TODO detailed description
        num_shots       Optional int, 0 by default
                        #TODO description/remove
    """

    def __init__(
        self,
        model_path,
        file_path: str,
        task_type: str,
        use_llmaaj: bool = None,
        output_dir: str = "eval_output",
    ):
        self.task_type = TaskType.from_string(task_type)
        self.file_path, file_columns = self._validate_file(file_path)
        self.user_card_data = UserUnitxtCardData(
            task_type=self.task_type, use_llmaaj=use_llmaaj, file_columns=file_columns
        )
        user_task = self._create_user_unitxt_card(output_dir)
        unitxt_recipe = self.get_recipe(user_task)
        # possibly inherit from mmlubranch and provide task dir and task name
        super().__init__(
            model_path=model_path, unitxt_recipe=unitxt_recipe, output_dir=output_dir
        )

    def get_recipe(self, user_task):
        return f"card={user_task}"

    def _create_user_unitxt_card(self, output_dir):
        user_task, user_task_dir = self._assign_task_name_and_dir(output_dir)
        os.makedirs(name=user_task_dir, exist_ok=True)
        os.environ["UNITXT_ARTIFACTORIES"] = user_task_dir
        card_file = os.path.join(user_task_dir, f"{user_task}.json")
        num_demos = 0  # TODO parameter? multiple options?
        data = {
            "__type__": "standard_recipe",
            "demos_pool_size": 20,  # TODO what should be the value here?
            "num_demos": num_demos,
            "demos_taken_from": "test",
            "max_train_instances": 1000,  # TODO what should be the value here?
            "max_validation_instances": 1000,  # TODO what should be the value here?
            "max_test_instances": 100,  # TODO what should be the value here?
            "card": {
                "loader": {"__type__": "load_csv", "files": {"test": self.file_path}},
                "task": {
                    "__type__": "task",
                    "input_fields": self.user_card_data.input_fields,
                    "reference_fields": {
                        "gt_output": "str"  # TODO should it be here regardless if it is available?
                    },
                    "metrics": [self.user_card_data.metric],
                    "prediction_type": "str",
                },
            },
            "template": self.user_card_data.templates,
        }
        with open(card_file, "w") as f:
            jsoned_data = json.dumps(data, indent=4)
            f.write(jsoned_data)
            f.write("\n")

        logger.debug("user unitxt card written to %s", card_file)
        return user_task

    def _validate_file(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        if not file_path.lower().endswith(".csv"):
            raise ValueError(f"The file '{file_path}' is not a CSV file.")
        required_fields = self.task_type.get_required_fields()
        df = pd.read_csv(file_path)
        file_columns = [col for col in df.columns if not df[col].isna().all()]
        if not all([col in file_columns for col in required_fields]):
            raise ValueError(
                f"Required columns missing from file. Task {self.task_type.name.lower()} requires the following columns: {required_fields}"
            )
        return file_path, file_columns
        # TODO if allowing few shot, make sure there are enough samples
