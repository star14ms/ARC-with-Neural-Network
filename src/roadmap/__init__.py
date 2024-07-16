import os

# Render the graph to a file (PDF)
base_dir = './output/graph/'
os.makedirs(base_dir, exist_ok=True)


def create_subgraph(graph, name=None, label='', color=None, bgcolor=None, nodes=[], edges=[], subgraph={}):
    with graph.subgraph(name=name) as c:
        c.attr(label=label, color=color, bgcolor=bgcolor)

        for node in nodes:
            c.node(**node)

        if subgraph:
            create_subgraph(c, **subgraph)

        # Connect the classifier node to the main yellow node
        for edge in edges:
            c.edge(edge['from'], edge['to'])


def create_record_label(rows):
    """
    Create a record label string for a Graphviz node.

    Parameters:
    rows (list of lists): Each inner list represents a row in the record. Each element in the inner list is a cell in that row.
                          To specify a port, use a tuple with (port_name, cell_content).

    Returns:
    str: A string representing the record label for Graphviz.
    """
    record = []
    for row in rows:
        cells = []
        for cell in row:
            if isinstance(cell, tuple):
                port, content = cell
                cells.append(f'<{port}> {content}')
            else:
                cells.append(cell)
        record.append(" | ".join(cells))
    return " | ".join(record)
