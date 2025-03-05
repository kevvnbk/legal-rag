import logging
import os
import pandas as pd
from datasets import load_dataset

import legalbench.utils
from legalbench.tasks import *

logger = logging.getLogger(__name__)

class DataUtils:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data = None # Lazy loading

        logger.info(f'Loading dataset: {dataset_name}')

    def get_data(self):
        """
        Get the loaded data.

        :return: Pandas DataFrame containing the loaded data.
        """
        if self.data is None:
            self.data = self._load_data()
            if self.data is not None:
                logger.info(f'Total data loaded: {len(self.data)} rows')
            else:
                logger.warning("No data was loaded.")
        return self.data

class LegalBench(DataUtils):
    def __init__(self, dataset_name, tasks=None, split=None):
        """
        Initialize LegalBench dataset.

        :param dataset_name: Name of the dataset.
        :param tasks: List of tasks to load.
        :param split: Split of the dataset to load.
        """
        super().__init__(dataset_name)
        self.task = tasks if tasks else self._get_all_tasks()
        self.split = split
        self.data = {}
        self.prompts = {}

        logger.info(f'Initializing dataset: {dataset_name}, task: {tasks}, split: {split}')

    def _get_all_tasks(self):
        """
        Get all tasks available in the dataset.

        :return: List of tasks available in the dataset.
        """
        return TASKS

    def _load_data(self, task):
        """
        Load dataset from LegalBench.

        :return: Loaded dataset as a Pandas DataFrame.
        """
        try: 
            dataset = load_dataset("nguha/legalbench", task)
            return dataset[self.split].to_pandas()
        except Exception as e:
            logger.error(f"Error loading dataset for task {task}: {e}")
            return None
    
    def _load_prompt_template(self, task):
        """
        Load the base prompt template for a given task.

        :param task: Task for which to load the prompt template.
        :return: The prompt template string.
        """
        task_dir = os.path.join("legalbench/tasks", task)
        prompt_path = os.path.join(task_dir, "base_prompt.txt")

        if not os.path.exists(prompt_path):
            logger.warning(f"Prompt template not found for task: {task}")
            return None
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
            
    def get_data(self):
        """
        Get the loaded data.

        :return: A dictionary of Pandas DataFrames for each task.
        """
        if not self.task:
            logger.error("No tasks specified or found in LegalBench.")
            raise ValueError("No tasks specified or found in LegalBench.")
    
        for task in self.task:
            if task not in self.data: # Lazy loading
                self.data[task] = self._load_data(task)

        return self.data if len(self.task) > 1 else self.data.get(self.task[0], None)
    
    def create_prompts(self, task, data_df):
        """
        Generate prompts from a prompt template and data.
        
        :param prompt_template: Template for the prompt.
        :param data_df: Data to use for generating prompts.

        :return: List of prompts generated from the template and data.
        """
        if task not in self.prompts:
            self.prompts[task] = self._load_prompt_template(task)

        prompt_template = self.prompts[task]
        if not prompt_template:
            logger.error(f"No prompt template found for task: {task}")
            return None

        return legalbench.utils.generate_prompts(prompt_template=prompt_template, data_df=data_df)

def load_data(dataset_name, tasks=None, split=None):
    """
    Load dataset from Hugging Face.
    
    :return: Loaded dataset as a Pandas DataFrame.
    """
    if dataset_name == 'legalbench':
        return LegalBench(dataset_name, tasks=tasks, split=split)