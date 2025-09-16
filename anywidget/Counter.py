import marimo

__generated_with = "0.15.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        [![Run Jupyter Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RichardPotthoff/Notebooks/main?filepath=anywidget/Counter.ipynb)   <- click here to open this file in MyBinder
    
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RichardPotthoff/Notebooks/blob/main/anywidget/Counter.ipynb)   <- click here to open this file in Google Colab
        """
    )
    return


@app.cell
def _():
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
    return (w,)


@app.cell
def _(w):
    observed=None
    def observer(e):
        global observed
        observed=e
        return
    w.observe(observer,["value"],"change")
    return (observed,)


@app.cell
def _(w):
    w.value=30
    return


@app.cell
def _(observed):
    observed
    return


@app.cell
def _():
    import os
    os.environ
    return (os,)


@app.cell
def _(os):
    import sys
    os.chdir('/var/mobile/Containers/Data/Application/3BA2F405-1E4D-41D1-8E03-EDDAA7B8A555/Documents/.jupyter')
    os.listdir()
    return


@app.cell
def _(os):
    os.listdir('serverconfig')
    return


@app.cell
def _():
    s=open('serverconfig/jupyterlabapputilsextensionannouncements.json').read()
    return (s,)


@app.cell
def _(s):
    s
    return


@app.cell
def _(os):
    os.listdir()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
