---
{marimo-version: 0.15.0, title: Counter}
---

[![Run Jupyter Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RichardPotthoff/Notebooks/main?filepath=anywidget/Counter.ipynb)   <- click here to open this file in MyBinder

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RichardPotthoff/Notebooks/blob/main/anywidget/Counter.ipynb)   <- click here to open this file in Google Colab

```python {.marimo}
import anywidget
import traitlets


class CounterWidget(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      let count = () => model.get("value");
      let btn = document.createElement("button");
      btn.classList.add("counter-button");
      btn.innerHTML = `count is ${count()}`;
      btn.addEventListener("click", () => {
        model.set("value", count() + 1);
        model.save_changes();
      });
      model.on("change:value", () => {
        btn.innerHTML = `count is ${count()}`;
      });
      el.appendChild(btn);
    }
    export default { render };
    """
    _css = """
    .counter-button {
      background-image: linear-gradient(to right, #a1c4fd, #a5e9fb);
      border: 0;
      border-radius: 10px;
      padding: 10px 50px;
      color: black;
    }
    """
    value = traitlets.Int(0).tag(sync=True)


w = CounterWidget()
w.value = 60

w
```

```python {.marimo}
observed=None
def observer(e):
    global observed
    observed=e
    return
w.observe(observer,["value"],"change")
```

```python {.marimo}
w.value=30
```

```python {.marimo}
observed
```

```python {.marimo}
import os
os.environ
```

```python {.marimo}
import sys
os.chdir('/var/mobile/Containers/Data/Application/3BA2F405-1E4D-41D1-8E03-EDDAA7B8A555/Documents/.jupyter')
os.listdir()
```

```python {.marimo}
os.listdir('serverconfig')
```

```python {.marimo}
s=open('serverconfig/jupyterlabapputilsextensionannouncements.json').read()
```

```python {.marimo}
s
```

```python {.marimo}
os.listdir()
```

```python {.marimo}
import marimo as mo
```