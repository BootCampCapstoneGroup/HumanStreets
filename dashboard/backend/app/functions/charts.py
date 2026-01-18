from typing import List, Dict, Any, Optional

def generate_chart_config(
    chart_type: str,
    title: str,
    x_data: List[Any],
    y_data: List[Any],
    x_label: str = "X Axis",
    y_label: str = "Y Axis",
    series_name: str = "Series 1",
    color: str = "#4a6cf7"
) -> Dict[str, Any]:
    """
    Generates a deterministic Plotly JSON configuration based on parameters.
    This acts as the 'Function' the user requested to ensure valid outputs.
    """
    
    # Validation: Ensure we have data
    if not x_data or not y_data:
        x_data = ["Category A", "Category B", "Category C"]
        y_data = [30, 50, 20]
        if title == "Chart":
            title = "Chart (Simulation Data)"

    
    chart_type = chart_type.lower()
    
    # Common Layout
    layout = {
        "title": title,
        "xaxis": {"title": x_label},
        "yaxis": {"title": y_label},
        "hovermode": "closest"
    }

    # Chart Specific Configuration
    data = []
    
    if chart_type in ["bar", "column"]:
        data.append({
            "type": "bar",
            "x": x_data,
            "y": y_data,
            "name": series_name,
            "marker": {"color": color}
        })
        
    elif chart_type in ["scatter", "line"]:
        data.append({
            "type": "scatter",
            "mode": "lines+markers" if chart_type == "line" else "markers",
            "x": x_data,
            "y": y_data,
            "name": series_name,
            "marker": {"color": color, "size": 10},
            "line": {"width": 3} if chart_type == "line" else None
        })
        
    elif chart_type == "pie":
        data.append({
            "type": "pie",
            "labels": x_data,
            "values": y_data,
            "hoverinfo": "label+percent+name"
        })
        # Pie charts don't need XY axes usually
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        
    else:
        # Default fallback to bar
        data.append({
            "type": "bar",
            "x": x_data,
            "y": y_data,
            "name": series_name
        })

    return {
        "data": data,
        "layout": layout
    }
