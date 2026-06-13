import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def load_jsonl(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def plot_scores(jsonl_file_path, output_file=None):
    """
    Plot usefulness and reliability scores for each source across queries.
    
    Args:
        jsonl_file_path: Path to the JSONL file
        output_file: Optional path to save the HTML file (default: same as input with .html extension)
    """
    # Load data
    data = load_jsonl(jsonl_file_path)
    
    # Extract all unique sources
    sources = set()
    for entry in data:
        sources.update(entry.get('usefulness_scores', {}).keys())
        sources.update(entry.get('reliability_scores', {}).keys())
    sources = sorted(list(sources))
    
    # Extract scores for each source and track which sources are ranked
    usefulness_data = {source: [] for source in sources}
    reliability_data = {source: [] for source in sources}
    ranked_sources = {source: [] for source in sources}  # Track if source is in ranks for each query
    query_ids = []
    query_metadata = []  # Store metadata for each query (question, answers, etc.)
    
    for entry in data:
        qid = entry.get('qid')
        if qid is None:
            continue
        
        query_ids.append(qid)
        usefulness_scores = entry.get('usefulness_scores', {})
        reliability_scores = entry.get('reliability_scores', {})
        ranks = entry.get('ranks', [])
        ranks_set = set(ranks)  # Convert to set for faster lookup
        
        # Store metadata for this query
        metadata = {
            'question': entry.get('question', ''),
            'answers': entry.get('answers', []),
            'response': entry.get('response', ''),
            'sampled_retrievers': entry.get('sampled_retrievers', []),
            'importance_score': entry.get('importance_score', []),
            'correctness': entry.get('correctness', None)
        }
        query_metadata.append(metadata)
        
        for source in sources:
            usefulness_data[source].append(usefulness_scores.get(source, None))
            reliability_data[source].append(reliability_scores.get(source, None))
            ranked_sources[source].append(source in ranks_set)  # Track if source is ranked
    
    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Usefulness Scores', 'Reliability Scores'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Color palette for different sources
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
    ]
    
    # Plot usefulness scores
    for idx, source in enumerate(sources):
        # Shorten source name for legend
        source_display = source.replace('experiments_dfnq_decentralized_polluted_invariant.', 'sources_')
        color = colors[idx % len(colors)]
        
        # Filter points: only show markers for ranked sources
        # Create separate arrays for markers (only ranked points) and lines (all points)
        marker_x = []
        marker_y = []
        hover_texts = []
        for i, (qid, score, ranked) in enumerate(zip(query_ids, usefulness_data[source], ranked_sources[source])):
            if ranked and score is not None:
                marker_x.append(qid)
                marker_y.append(score)
                
                # Create hover text with metadata
                metadata = query_metadata[i]
                hover_lines = []
                hover_lines.append(f"<b>Query ID:</b> {qid}")
                hover_lines.append(f"<b>Question:</b> {metadata['question']}")
                
                # Format answers
                answers_str = ', '.join(metadata['answers']) if metadata['answers'] else 'N/A'
                hover_lines.append(f"<b>Answers:</b> {answers_str}")
                
                # Format response (truncate if too long)
                response = metadata['response'].strip()
                if len(response) > 100:
                    response = response[:100] + "..."
                hover_lines.append(f"<b>Response:</b> {response}")
                
                # Format all importance_score entries as JSON
                importance_scores = metadata['importance_score']
                if importance_scores:
                    # Convert to a more readable format: shorten source names and format with line breaks
                    formatted_items = []
                    for item in importance_scores:
                        if isinstance(item, list) and len(item) == 2:
                            source_name = item[0].replace('experiments_dfnq_decentralized_polluted_invariant.', 'sources_')
                            formatted_items.append(f"  [\"{source_name}\", {item[1]:.6f}]")
                    importance_json = "[\n" + ",\n".join(formatted_items) + "\n]"
                    # Replace newlines with <br> for HTML display
                    importance_json_html = importance_json.replace('\n', '<br>')
                    hover_lines.append(f"<b>Importance Scores:</b><br>{importance_json_html}")
                else:
                    hover_lines.append(f"<b>Importance Scores:</b> N/A")
                
                # Format correctness
                correctness = metadata['correctness']
                correctness_str = 'True' if correctness is True else 'False' if correctness is False else 'N/A'
                hover_lines.append(f"<b>Correctness:</b> {correctness_str}")
                
                hover_texts.append('<br>'.join(hover_lines))
        
        # Plot line with all data points
        fig.add_trace(
            go.Scatter(
                x=query_ids,
                y=usefulness_data[source],
                mode='lines',
                name=source_display,
                line=dict(color=color, width=2),
                legendgroup=source_display,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Plot markers only for ranked points
        if marker_x:  # Only add if there are ranked points
            fig.add_trace(
                go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode='markers',
                    name=source_display,
                    marker=dict(
                        size=8, 
                        symbol='circle', 
                        color=color,
                        sizemode='diameter'
                    ),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup=source_display,
                    showlegend=False  # Don't duplicate in legend
                ),
                row=1, col=1
            )
    
    # Plot reliability scores
    for idx, source in enumerate(sources):
        source_display = source.replace('experiments_dfnq_decentralized_polluted_invariant.', 'sources_')
        color = colors[idx % len(colors)]
        
        # Filter points: only show markers for ranked sources
        marker_x = []
        marker_y = []
        hover_texts = []
        for i, (qid, score, ranked) in enumerate(zip(query_ids, reliability_data[source], ranked_sources[source])):
            if ranked and score is not None:
                marker_x.append(qid)
                marker_y.append(score)
                
                # Create hover text with metadata
                metadata = query_metadata[i]
                hover_lines = []
                hover_lines.append(f"<b>Query ID:</b> {qid}")
                hover_lines.append(f"<b>Question:</b> {metadata['question']}")
                
                # Format answers
                answers_str = ', '.join(metadata['answers']) if metadata['answers'] else 'N/A'
                hover_lines.append(f"<b>Answers:</b> {answers_str}")
                
                # Format response (truncate if too long)
                response = metadata['response'].strip()
                if len(response) > 100:
                    response = response[:100] + "..."
                hover_lines.append(f"<b>Response:</b> {response}")
                
                # Format all importance_score entries as JSON
                importance_scores = metadata['importance_score']
                if importance_scores:
                    # Convert to a more readable format: shorten source names and format with line breaks
                    formatted_items = []
                    for item in importance_scores:
                        if isinstance(item, list) and len(item) == 2:
                            source_name = item[0].replace('experiments_dfnq_decentralized_polluted_invariant.', 'sources_')
                            formatted_items.append(f"  [\"{source_name}\", {item[1]:.6f}]")
                    importance_json = "[\n" + ",\n".join(formatted_items) + "\n]"
                    # Replace newlines with <br> for HTML display
                    importance_json_html = importance_json.replace('\n', '<br>')
                    hover_lines.append(f"<b>Importance Scores:</b><br>{importance_json_html}")
                else:
                    hover_lines.append(f"<b>Importance Scores:</b> N/A")
                
                # Format correctness
                correctness = metadata['correctness']
                correctness_str = 'True' if correctness is True else 'False' if correctness is False else 'N/A'
                hover_lines.append(f"<b>Correctness:</b> {correctness_str}")
                
                hover_texts.append('<br>'.join(hover_lines))
        
        # Plot line with all data points
        fig.add_trace(
            go.Scatter(
                x=query_ids,
                y=reliability_data[source],
                mode='lines',
                name=source_display,
                line=dict(color=color, width=2),
                legendgroup=source_display,
                showlegend=False  # Hide duplicate legend entries
            ),
            row=2, col=1
        )
        
        # Plot markers only for ranked points
        if marker_x:  # Only add if there are ranked points
            fig.add_trace(
                go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode='markers',
                    name=source_display,
                    marker=dict(
                        size=8, 
                        symbol='square', 
                        color=color,
                        sizemode='diameter'
                    ),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup=source_display,
                    showlegend=False  # Don't duplicate in legend
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_xaxes(title_text="Query ID", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    
    fig.update_layout(
        title_text="Source Scores Over Queries",
        title_x=0.5,
        height=800,
        hovermode='closest',  # Changed from 'x unified' to show individual point hovers
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title_text="Sources"
        )
    )
    
    # Save or show
    if output_file is None:
        output_file = jsonl_file_path.replace('.jsonl', '_scores.html')
    
    # fig.show()
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")
    
    return fig

if __name__ == "__main__":
    # Default to the JSONL file in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_file = os.path.join(script_dir, "dRAG_polluted_token_llama32-3B.jsonl")
    
    # Check if file exists
    if os.path.exists(jsonl_file):
        plot_scores(jsonl_file)
    else:
        print(f"File not found: {jsonl_file}")
        print("Please provide the path to your JSONL file:")
        print("Usage: plot_scores('path/to/your/file.jsonl')")

