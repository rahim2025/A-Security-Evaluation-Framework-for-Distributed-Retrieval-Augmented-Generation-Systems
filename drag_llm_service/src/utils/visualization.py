# visualization for the logs based on the local log files
# TODO: wenyi - api-based visualization

import plotly.graph_objects as go
import json
import os

def visualize_reliability_history_local(drag_log_file: str, source_path = "data/unreliable_large_source", save_path = "demo_figs/"):

    folder_name = os.path.basename(source_path)
    
    source_names = []
    for file in os.listdir(source_path):
        source_path_log_prefix = folder_name + "-" + file.split('.')[0]
        source_names.append(source_path_log_prefix)
    source_names = sorted(source_names, key=lambda x: int(x.split('_')[-1]))

    reliability_history_data = {source: [100] for source in source_names}
    feed_back_data = {source: ['Initialization'] for source in source_names}
    x_positions = {source: [0] for source in source_names}
    feeb_back_count = 0

    with open(drag_log_file, 'r') as file:
        for line in file:
            log_record = json.loads(line)
            if log_record['operation'] == 'post__create-feedback-record':
                feeb_back_count += 1
                info_json = json.loads(log_record['record']['input'])
                info_json['response'] = log_record['record']['output'].replace('\\n', '')
                info_json.update(json.loads(log_record['record']['reserved']))
                for source in json.loads(log_record['record']['inputFrom']):
                    x_positions[source].append(feeb_back_count)
                    feed_back_data[source].append(json.dumps(info_json, indent=2).replace('\n', '<br>'))
            elif log_record['operation'].startswith('put__update-reliability-record_'):
                current_source = log_record['operation'].split('put__update-reliability-record_')[-1]
                reliability_history_data[current_source].append(log_record['record']['reliabilityScore'])
            else:
                continue

    fig2 = go.Figure()

    for source in source_names:
        fig2.add_trace(go.Scatter(
            x=x_positions[source],
            y=reliability_history_data[source],
            mode='markers+lines',
            name=source,
            hovertext=feed_back_data[source],
        ))

    file_name = drag_log_file.split('/')[-2] + "_reliability_history.html"
    fig2.update_layout(
        title="Reliability Scores History for " + file_name,
        xaxis_title="Queries",
        yaxis_title="Reliability Scores",
        showlegend=True
    )

    fig2.write_html(save_path + file_name)
    fig2.show()


def visualize_accumulated_chosen_times_local(drag_log_file: str, source_path = "data/unreliable_large_source", save_path = "demo_figs/"):
    folder_name = os.path.basename(source_path)
    
    source_names = []
    for file in os.listdir(source_path):
        source_path_log_prefix = folder_name + "-" + file.split('.')[0]
        source_names.append(source_path_log_prefix)
    source_names = sorted(source_names, key=lambda x: int(x.split('_')[-1]))

    accumulated_chosen_times_data = {source: [0] for source in source_names}
    feed_back_data = {source: ['Initialization'] for source in source_names}
    x_positions = [0]
    feedback_count = 0

    with open(drag_log_file, 'r') as file:
        for line in file:
            log_record = json.loads(line)
            if log_record['operation'] == 'post__create-feedback-record':
                feedback_count += 1
                x_positions.append(x_positions[-1] + 1)
                info_json = json.loads(log_record['record']['input'])
                info_json['response'] = log_record['record']['output'].replace('\\n', '')
                info_json.update(json.loads(log_record['record']['reserved']))
                for source in accumulated_chosen_times_data.keys():
                    accumulated_chosen_times_data[source].append(accumulated_chosen_times_data[source][-1])
                    feed_back_data[source].append('Not Chosen')
                for source in json.loads(log_record['record']['inputFrom']):
                    accumulated_chosen_times_data[source][-1] += 1
                    feed_back_data[source][-1] = json.dumps(info_json, indent=2).replace('\n', '<br>')
            else:
                continue

    fig2 = go.Figure()

    for source in source_names:
        fig2.add_trace(go.Scatter(
            x=x_positions,
            y=accumulated_chosen_times_data[source],
            mode='markers+lines',
            name=source,
            hovertext=feed_back_data[source],
        ))

    file_name = drag_log_file.split('/')[-2] + "_accumulated_chosen_times.html"
    fig2.update_layout(
        title="Accumulated Chosen Times for " + file_name,
        xaxis_title="Queries",
        yaxis_title="Accumulated Chosen Times",
        showlegend=True
    )

    fig2.write_html(save_path + file_name)
    fig2.show()