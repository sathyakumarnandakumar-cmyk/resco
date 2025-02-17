import os

import neptune.new as neptune
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.oauth2.service_account import Credentials
import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


VALIDATION_FACTOR = 0.7


def get_google_credentials() -> Credentials:
    return Credentials.from_service_account_file(
        filename=os.environ.get("CREDENTIALS_FILE"),
        scopes=os.environ.get("SCOPES").split(","),
    )


def authenticate_google_service(credentials: Credentials) -> gspread.Client:
    return gspread.authorize(credentials)


def get_google_spreadsheet(client: gspread.client.Client) -> gspread.spreadsheet:
    spreadsheet = client.open(os.environ.get("SPREADSHEET_NAME"))
    return spreadsheet.sheet1


def get_last_spreadsheet_experiment_id(worksheet: gspread.worksheet) -> str:
    return worksheet.get(range_name="A2")[0][0]


def get_neptune_experiment_ids() -> list:
    project = neptune.init_project(mode="read-only")
    neptune_experiments_names = project.fetch_runs_table(
        columns="sys/id", state="inactive"
    ).to_pandas()
    list_of_neptune_experiments_names = neptune_experiments_names["sys/id"].to_list()
    return sorted(
        list_of_neptune_experiments_names,
        key=lambda x: int(x.split("-")[-1]),
        reverse=True,
    )


def build_neptune_run_data(experiment_id: str, run: neptune.Run) -> dict:
    data = {
        "ID": experiment_id,
        "Model": run["parameters/algorithm"].fetch(),
        "Reward": run["parameters/reward"].fetch(),
        "Action_frequency": f"action every {run['parameters/action_frequency'].fetch()} seconds",
        "Date": "26-11-2022",
        "Additional tags": run["sys/tags"].fetch(),
        "number_of_training_episodes": run[
            "parameters/number_of_training_episodes"
        ].fetch(),
        "number_of_validation_episodes": run[
            "parameters/number_of_validation_episodes"
        ].fetch(),
    }
    return data


def get_neptune_metrics(run: neptune.Run) -> dict:
    metrics = {
        "total_average_delays_of_all_vehicles_from_all_routes": "metrics/validation/total_average_delays_of_all_vehicles_from_all_routes",
        "total_average_delays_real_times_by_ideal_times": "metrics/validation/total_average_delays_real_times_by_ideal_times",
        "total_sum_delays_of_all_vehicles_from_all_routes": "metrics/validation/total_sum_delays_of_all_vehicles_from_all_routes",
        "count_of_vehicles_completing_journey": "metrics/validation/count_of_vehicles_completing_journey",
        "total_waiting_time_on_the_incoming_lanes_in_episode": "metrics/validation/total_waiting_time_on_the_incoming_lanes_in_episode",
        "total_waiting_time_on_the_incoming_lanes_in_episode2": "metrics/validation/total_waiting_time_on_the_incoming_lanes_in_episode2",
        "total_time_of_journey": "metrics/validation/total_time_of_journey",
        "average_time_of_journey": "metrics/validation/average_time_of_journey",
        "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey": "metrics/validation/total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey",
    }

    metrics_data = {}
    for metric_name, metric_path in metrics.items():
        metric_values = run[metric_path].fetch_values().value.values
        metrics_data[f"{metric_name}_first"] = round(metric_values[0], 5)
        metrics_data[f"{metric_name}_last"] = round(metric_values[-1], 5)
        if metric_name == "count_of_vehicles_completing_journey":
            metrics_data[f"{metric_name}_max"] = round(max(metric_values), 5)
        else:
            metrics_data[f"{metric_name}_min"] = round(min(metric_values), 5)

    return metrics_data


def fetch_neptune_run_data(experiment_id: str) -> dict | int:
    run = neptune.init_run(with_id=experiment_id)
    data = build_neptune_run_data(experiment_id, run)
    metrics = get_neptune_metrics(run)
    data.update(metrics)

    number_of_completed_validation_episodes = len(
        run["metrics/validation/count_of_vehicles_completing_journey"]
        .fetch_values()
        .value.values
    )
    return data, number_of_completed_validation_episodes


def update_spreadsheet_with_neptune_data(
    worksheet: gspread.worksheet, data_from_neptune: dict
) -> None:
    df_new_data = pd.DataFrame(data_from_neptune)
    df_existing_spreadsheet = get_as_dataframe(worksheet, evaluate_formulas=True)
    df_existing_spreadsheet.columns = df_new_data.columns
    updated_df = pd.concat([df_new_data, df_existing_spreadsheet], ignore_index=True)
    set_with_dataframe(worksheet, updated_df)


def process_neptune_experiments(
    worksheet: gspread.worksheet, last_spreadsheet_experiment_id: str
) -> None:
    experiment_ids = get_neptune_experiment_ids()
    data_from_neptune = []

    for experiment_id in experiment_ids:
        run_data, number_of_completed_validation_episodes = fetch_neptune_run_data(
            experiment_id
        )
        number_of_expected_validation_episodes = run_data[
            "number_of_validation_episodes"
        ]

        if experiment_id == last_spreadsheet_experiment_id:
            break
        elif (
            number_of_completed_validation_episodes
            / number_of_expected_validation_episodes
        ) >= VALIDATION_FACTOR:
            data_from_neptune.append(run_data)

    update_spreadsheet_with_neptune_data(worksheet, data_from_neptune)


def main() -> None:
    credentials = get_google_credentials()
    google_client = authenticate_google_service(credentials)
    worksheet = get_google_spreadsheet(google_client)
    last_spreadsheet_experiment_id = get_last_spreadsheet_experiment_id(worksheet)
    process_neptune_experiments(worksheet, last_spreadsheet_experiment_id)


if __name__ == "__main__":
    main()
