"""This module contains a convenient function that returns the id of an experiment."""
import mlflow


def get_experiment_id(name: str):
    """Get the id of an experiment by name.

    If the experiment does not exist, it creates it and returns the id. If it does exist, the id is consulted and
    returned.

    :param name: Name of the experiment.
    :return: Id of the experiment.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(name)

    if experiment:
        return experiment.experiment_id
    else:
        return client.create_experiment(name)
