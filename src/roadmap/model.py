import graphviz
from __init__ import base_dir, create_subgraph, create_record_label


# Create a new directed graph
dot = graphviz.Digraph(comment='Model Instructions Flowchart')

# Define the subgraphs with shared arguments
subgraphs = [
    {
        'name': 'cluster_pattern',
        'label': '1. Extract Pattern of Data',
        'color': 'purple',
        'bgcolor': 'lavender',
        'nodes': [
            {'name': 'pattern_feature_extractor_input', 'label': 'Feature Extractor', 'shape': 'box3d', 'style': 'filled', 'fillcolor': 'yellow', 'bg_color': 'black'},
            {'name': 'pattern_feature_extractor_output', 'label': 'Feature Extractor', 'shape': 'box3d', 'style': 'filled', 'fillcolor': 'yellow', 'bg_color': 'black'},
            {'name': 'pattern_extractor', 'label': 'Pattern Extractor', 'shape': 'box', 'style': 'filled', 'color': 'yellow'},
        ],
        'subgraph': {
            'name': 'cluster_pattern_inside',
            'label': '',
            'bgcolor': 'white',
            'nodes': [
                {'name': 'pattern_feature_map_input', 'label': 'Feature Maps', 'shape': 'box3d', 'style': 'filled', 'fillcolor': 'lightgray', 'bg_color': 'black'},
                {'name': 'pattern_feature_map_output', 'label': 'Feature Maps', 'shape': 'box3d', 'style': 'filled', 'fillcolor': 'lightgray', 'bg_color': 'black'},
            ],
        },
        'edges': [
            {'from': 'pattern_feature_extractor_input', 'to': 'pattern_feature_map_input'},
            {'from': 'pattern_feature_extractor_output', 'to': 'pattern_feature_map_output'},
            {'from': 'pattern_feature_map_input', 'to': 'pattern_extractor'},
            {'from': 'pattern_feature_map_output', 'to': 'pattern_extractor'},
        ],
    },
    {
        'name': 'cluster_feature',
        'label': '2. Extract Feature of Data',
        'color': 'purple',
        'bgcolor': 'lavender',
        'nodes': [
            {'name': 'feature_extractor', 'label': 'Feature Extractor', 'shape': 'box', 'style': 'filled', 'color': 'yellow'},
        ],
        'edges': [],
    },
    {
        'name': 'cluster_shape',
        'label': '3. Determine Output Shape and Colors',
        'color': 'blue',
        'bgcolor': 'lightblue',
        'nodes': [
            {'name': 'shape_classifier', 'label': 'Shape Classifier', 'shape': 'box', 'style': 'filled', 'color': 'aqua'},
            {'name': 'shape_predictor', 'label': 'Shape Predictor', 'shape': 'box', 'style': 'filled', 'color': 'yellow'},
        ],
        'subgraph': {
            'name': 'cluster_shape_inside',
            'label': '',           
            'bgcolor': 'white',
            'nodes': [
                {'name': 'shape_classes', 'label': create_record_label([['Same as Input', 'Based on Input', 'Independent (Fixed)']]), 'shape': 'record', 'style': 'filled', 'fillcolor': 'lightgrey'},
            ],
        },
        'edges': [
            {'from': 'shape_classifier', 'to': 'shape_classes'},
            {'from': 'shape_classes', 'to': 'shape_predictor'},
        ],
    },
    {
        'name': 'cluster_canvas',
        'label': '4. Determine Canvas',
        'color': 'green',
        'bgcolor': 'lightgreen',
        'nodes': [
            {'name': 'canvas_classifier', 'label': 'Canvas Classifier', 'shape': 'box', 'style': 'filled', 'color': 'aqua'},
            {'name': 'canvas_generator', 'label': 'Canvas Generator', 'shape': 'box', 'style': 'filled', 'color': 'yellow'},
        ],
        'subgraph': {
            'name': 'cluster_canvas_inside',
            'label': '',
            'bgcolor': 'white',
            'nodes': [
                {'name': 'canvas_classes', 'label': create_record_label([['Starts from Empty', 'Starts from Input']]), 'shape': 'record', 'style': 'filled', 'fillcolor': 'lightgrey'},
            ],
        },
        'edges': [
            {'from': 'canvas_classifier', 'to': 'canvas_classes'},
            {'from': 'canvas_classes', 'to': 'canvas_generator'},
        ],
    },
    {
        'name': 'cluster_filler',
        'label': '5. Determine Each Pixel Value of Output',
        'color': 'red',
        'bgcolor': 'lightcoral',
        'nodes': [
            {'name': 'filler_classifier', 'label': 'Filler Classifier', 'shape': 'box', 'style': 'filled', 'color': 'aqua'},
            {'name': 'canvas_filler', 'label': 'Canvas Filler', 'shape': 'box', 'style': 'filled', 'color': 'yellow'},
        ],
        'subgraph': {
            'name': 'cluster_filler_inside',
            'label': '',
            'bgcolor': 'white',
            'nodes': [
                {'name': 'filler_classes', 'label': create_record_label([['<f0>No-Change', '<f1>Substitute']]), 'shape': 'record', 'style': 'filled', 'fillcolor': 'lightgray'},
            ],
        },
        'edges': [
            {'from': 'filler_classifier', 'to': 'filler_classes'},
            {'from': 'filler_classes:<f1>', 'to': 'canvas_filler'},
        ],
    }
]

# Create subgraphs
for subgraph in subgraphs:
    create_subgraph(dot, **subgraph)

# Define connections between the main components
dot.node('pattern', 'Pattern', shape='box', style='filled', color='lightgrey')
dot.node('feature_map_test', 'Feature Map', shape='box', style='filled', color='lightgrey')
dot.node('output_shape', 'Output Shape', shape='box', style='filled', color='lightgrey')
dot.node('canvas', 'Canvas', shape='box', style='filled', color='lightgrey')

dot.edge('pattern_extractor', 'pattern')
dot.edge('feature_extractor', 'feature_map_test')
dot.edge('pattern', 'shape_classifier')
dot.edge('pattern', 'canvas_classifier')
dot.edge('pattern', 'filler_classifier')
dot.edge('feature_map_test', 'shape_classifier')
dot.edge('feature_map_test', 'canvas_classifier')
dot.edge('feature_map_test', 'filler_classifier')
dot.edge('canvas_generator', 'canvas')
dot.edge('shape_predictor', 'output_shape')
dot.edge('canvas', 'filler_classifier')
dot.edge('output_shape', 'canvas_classifier')

# Define the input-output data nodes
dot.node('input_output', 'Input-Output Pairs', shape='box3d', style='filled', fillcolor='orange', bg_color='black')
dot.node('inputs', 'Inputs', shape='box3d', style='filled', fillcolor='orange', bg_color='black')
dot.node('outputs', 'Outputs', shape='box3d', style='filled', fillcolor='orange', bg_color='black')
dot.node('input', 'Input', shape='box', style='filled', color='orange')
dot.node('output', 'output', shape='box', style='filled', color='orange')

# Connect input-output data nodes to the classifiers
dot.edge('input_output', 'inputs')
dot.edge('input_output', 'outputs')
dot.edge('inputs', 'pattern_feature_extractor_input')
dot.edge('outputs', 'pattern_feature_extractor_output')
dot.edge('input', 'feature_extractor')
dot.edge('filler_classes:<f0>', 'output')
dot.edge('canvas_filler', 'output')

# Save and render the graph to a file (PDF or PNG)
dot.render(base_dir + 'model_instructions_flowchart', format='png', view=False)
