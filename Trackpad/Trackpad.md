---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

[![Run Jupyter Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RichardPotthoff/Notebooks/main?filepath=Trackpad/Trackpad.ipynb)   <- click here to open this file in MyBinder
    
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RichardPotthoff/Notebooks/blob/main/Trackpad/Trackpad.ipynb)   <- click here to open this file in Google Colab

```python
def capture_console(code):
    return (
"""
try{console._o=console._o||{log:console.log,error:console.error,warn:console.warn,info:console.info};[['log'],['error','red'],['warn'],['info']].forEach(([t,c="black"])=>{console[t]=console._o[t];let d=element.consoleOutput;if(!d){d=document.createElement("div");d.id="console-output";d.style.cssText="white-space:pre-wrap;font-family:monospace;padding:0px;background:#f0f0f0;line-height:1.1;";element.appendChild(d);element.consoleOutput=d;}let o=console[t],n=o;while(n&&n.toString().indexOf("[native code]")<0)n=n.apply?function(...a){return n.apply(this,a);}:null;o=function(...a){(n||console._o[t]).apply(console,a);let s=a.map(e=>typeof e==='object'?JSON.stringify(e,null,2):e).join(' ');const m=document.createElement("div");m.innerText=`[${t.toUpperCase()}] ${s}`;m.style.cssText=`margin:0;line-height:1.1;padding:0px 0;color:${c};`;d.appendChild(m);};console[t]=o;});
"""  +
code + 
"""
}catch(e){console.error("Error:",e);}finally{[['log'],['error','red'],['warn'],['info']].forEach(([t])=>console._o&&console._o[t]&&(console[t]=console._o[t]));}
"""
)
```

## 2-way synchronisation using text input ipywidgets  

```python
import ipywidgets as widgets
from IPython.display import display, Javascript
import uuid
import json

# Create unique UUIDs for widgets
config_uuid = "uuid" + str(uuid.uuid4()).replace('-', '')
valxy_uuid = "uuid" + str(uuid.uuid4()).replace('-', '')
x_slider_uuid = "uuid" + str(uuid.uuid4()).replace('-', '')
y_slider_uuid = "uuid" + str(uuid.uuid4()).replace('-', '')

# Configuration (width, height, minx, maxx, miny, maxy)
initial_config = {
    "width": 400,
    "height": 300,
    "minx": -40 ,
    "maxx": 40,
    "miny": -30,
    "maxy": 30
}

config_widget = widgets.Text(value=json.dumps(initial_config), layout={'display': 'none'})
config_widget.add_class(config_uuid)
minx,maxx,miny,maxy,x0,y0=(lambda minx,maxx,miny,maxy,**_:(minx,maxx,miny,maxy,(minx+maxx)/2,(miny+maxy)/2))(**initial_config)

# Current x/y coordinates (scaled to minx/maxx, miny/maxy)
initial_valxy = {"x": x0, "y": y0}
valxy_widget = widgets.Text(value=json.dumps(initial_valxy), layout={'display': 'none'})
valxy_widget.add_class(valxy_uuid)

# Sliders for demonstration (not part of Trackpad)
x_slider = widgets.FloatSlider(value=x0, min=minx, max=maxx, description="X:")
x_slider.add_class(x_slider_uuid) 
y_slider = widgets.FloatSlider(value=y0, min=miny, max=maxy, description="Y:")
y_slider.add_class(y_slider_uuid)

# Update sliders' ranges when config changes
def on_config_change(change):
    try:
        config = json.loads(change["new"])
        x_slider.min = config["minx"]
        x_slider.max = config["maxx"]
        y_slider.min = config["miny"]
        y_slider.max = config["maxy"]
        # Ensure current values are within new range
        valxy = json.loads(valxy_widget.value)
        valxy["x"] = max(config["minx"], min(config["maxx"], valxy["x"]))
        valxy["y"] = max(config["miny"], min(config["maxy"], valxy["y"]))
        valxy_widget.value = json.dumps(valxy)
    except (json.JSONDecodeError, KeyError):
        pass

config_widget.observe(on_config_change, names="value")

# Handle canvas updates (JavaScript → Python)
def on_valxy_change(change):
    try:
        valxy = json.loads(change["new"])
        x, y = valxy["x"], valxy["y"]
        # Update sliders
        x_slider.value = x
        y_slider.value = y
    except (json.JSONDecodeError, KeyError):
        pass

valxy_widget.observe(on_valxy_change, names="value")

# Handle slider updates (Python → JavaScript)
def on_x_slider_change(change):
    valxy = json.loads(valxy_widget.value)
    valxy["x"] = change["new"]
    valxy_widget.value = json.dumps(valxy)

def on_y_slider_change(change):
    valxy = json.loads(valxy_widget.value)
    valxy["y"] = change["new"]
    valxy_widget.value = json.dumps(valxy)

x_slider.observe(on_x_slider_change, names="value")
y_slider.observe(on_y_slider_change, names="value")

# Output widget for canvas
output = widgets.Output()

with output:
    display(Javascript(f"""
    // Read configuration
    const configWidget = document.querySelector('.{config_uuid}');
    let config = JSON.parse(configWidget.querySelector('input').value);
    const canvas = document.createElement('canvas');
    canvas.width = config.width;
    canvas.height = config.height;
    canvas.style.border = '1px solid black';
    element.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    // Trackpad state
    let isDragging = false;
    let currentX = 0;
    let currentY = 0;

    // Draw dot
    function drawDot(x, y) {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'blue';
        ctx.fill();
    }}

    // Scale coordinates
    function scaleToCanvas(x, y) {{
        const minx = config.minx, maxx = config.maxx;
        const miny = config.miny, maxy = config.maxy;
        const width = config.width, height = config.height;
        const canvasX = (x - minx) / (maxx - minx) * width;
        const canvasY = height - ((y - miny) / (maxy - miny) * height); // Invert Y-axis
        return [canvasX, canvasY];
    }}

    function scaleFromCanvas(canvasX, canvasY) {{
        const minx = config.minx, maxx = config.maxx;
        const miny = config.miny, maxy = config.maxy;
        const width = config.width, height = config.height;
        const x = minx + (canvasX / width) * (maxx - minx);
        const y = miny + ((height - canvasY) / height) * (maxy - miny); // Invert Y-axis
        return [x, y];
    }}

    // Event handlers
    function handleStart(x, y) {{
        isDragging = true;
        updatePosition(x, y);
    }}

    function handleMove(x, y) {{
        if (isDragging) {{
            updatePosition(x, y);
        }}
    }}

    function handleEnd() {{
        isDragging = false;
    }}

    function updatePosition(x, y) {{
        currentX = Math.max(0, Math.min(x, canvas.width));
        currentY = Math.max(0, Math.min(y, canvas.height));
        drawDot(currentX, currentY);
        const [scaledX, scaledY] = scaleFromCanvas(currentX, currentY);
        const valxyWidget = document.querySelector('.{valxy_uuid}');
        if (valxyWidget) {{
            const input = valxyWidget.querySelector('input');
            input.value = JSON.stringify({{ x: scaledX, y: scaledY }});
            const changeEvent = new Event('change', {{ bubbles: true }});
            input.dispatchEvent(changeEvent);
            const inputEvent = new Event('input', {{ bubbles: true }});
            input.dispatchEvent(inputEvent);
        }}
    }}

    // Mouse events
    canvas.addEventListener('mousedown', (e) => {{
        const rect = canvas.getBoundingClientRect();
        handleStart(e.clientX - rect.left, e.clientY - rect.top);
    }});

    canvas.addEventListener('mousemove', (e) => {{
        const rect = canvas.getBoundingClientRect();
        handleMove(e.clientX - rect.left, e.clientY - rect.top);
    }});

    canvas.addEventListener('mouseup', handleEnd);
    canvas.addEventListener('mouseleave', handleEnd);

    // Touch events
    canvas.addEventListener('touchstart', (e) => {{
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        handleStart(touch.clientX - rect.left, touch.clientY - rect.top);
    }});

    canvas.addEventListener('touchmove', (e) => {{
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        handleMove(touch.clientX - rect.left, touch.clientY - rect.top);
    }});

    canvas.addEventListener('touchend', handleEnd);
    canvas.addEventListener('touchcancel', handleEnd);

    // Initial draw
    const valxyWidget = document.querySelector('.{valxy_uuid}');
    let lastValxy = valxyWidget.querySelector('input').value;
    let [x, y] = scaleToCanvas({initial_valxy["x"]}, {initial_valxy["y"]});
    drawDot(x, y);

    // Poll for Python → JavaScript updates
    setInterval(() => {{
        // Update config
        const newConfig = JSON.parse(configWidget.querySelector('input').value);
        if (canvas.width !== newConfig.width || canvas.height !== newConfig.height) {{
            canvas.width = newConfig.width;
            canvas.height = newConfig.height;
        }}
        config = newConfig; // Update config for scaling
        const newValxy = valxyWidget.querySelector('input').value;
        if (newValxy !== lastValxy) {{
            lastValxy = newValxy;
            const valxy = JSON.parse(newValxy);
            [currentX, currentY] = scaleToCanvas(valxy.x, valxy.y);
            drawDot(currentX, currentY);
        }}
    }}, 100);
    """))

# Display widgets
display(config_widget, valxy_widget, x_slider, y_slider, output)
```

```python
minx,maxx,miny,maxy,x0,y0=(lambda minx,maxx,miny,maxy,**_:(minx,maxx,miny,maxy,(minx+maxx)/2,(miny+maxy)/2))(**initial_config)
print(minx,maxx,miny,maxy,x0,y0)
```

```python

```
