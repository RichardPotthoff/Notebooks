import marimo

__generated_with = "0.13.10"
app = marimo.App()


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
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
