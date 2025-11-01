import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", auto_download=["ipynb"])

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import anywidget
    import traitlets
    import json
    from typing import Callable, Optional


@app.class_definition
class SIUnitInputAnywidget(anywidget.AnyWidget):
    """Custom widget for unit-aware numeric input. .si_value is always in SI units."""
    # Base JS module (ESM) template with escaped braces and placeholder for conversions

    # Minimal traits (only essential sync)
    si_value = traitlets.Float(float('nan')).tag(sync=True)  # SI value (reactive)
    defaults = traitlets.Dict({}).tag(sync=True)
    editable = traitlets.Bool(True).tag(sync=True)
    _esm = (r"""
     //alert("executing _esm body"); 
     //console.log("executing _esm body");
     // Format display value (basic Python-like support)
    function formatValue(spec, val) {
        try {
            const fixedMatch = spec.match(/\.(\d+)f/);
            const expMatch = spec.match(/\.(\d+)e/);
            const precMatch = spec.match(/\.(\d+)g/);
            if (expMatch) {
                const digits = parseInt(expMatch[1]) ?? 2;
                return val.toExponential(digits);
            } else if (precMatch) {
                const digits = parseInt(precMatch[1]) ?? 2;
                return val.toPrecision(digits);
            }
            const digits = parseInt(fixedMatch ? fixedMatch[1] : 2) ?? 2;
            return val.toFixed(digits);
        } catch {
            return val.toFixed(2);
        }
    }
        // Dynamic conversions from dict (m * display + b for toSI; inverse for fromSI)
    function toSI(val, unit, conv) {
        const [m, b = 0] = Array.isArray(conv[unit]) ? conv[unit] : [conv[unit], 0];
        return m * val + b;
    }

    function fromSI(val, unit, conv) {
        const [m, b = 0] = Array.isArray(conv[unit]) ? conv[unit] : [conv[unit], 0];
        if (m === 0) return val;  // Avoid div/0
        return (val - b) / m;
    }

    function initialize({model}) {
       return ()=>{
           //clean up event listeners
           }   
    }

    function render({ model, el }) {
        let formatSpec  = JSON.parse(getComputedStyle(el).getPropertyValue('--format-spec').trim() || 'null');
        let model_defaults = model.get('defaults');
        let format_spec = formatSpec || model_defaults.default_format_spec || '.2f';
        let unit_conversions = model_defaults.unit_conversions;
        let current_unit = model_defaults.default_unit || Object.keys(unit_conversions)[0];
        let current_si = model.get('si_value');
        let editable = model.get("editable");
        // Create UI elements
        const container = document.createElement('div');
        container.className= 'si-container';

        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = editable ? 'Enter value' : '';
        function update_input_text(){
            input.value = formatValue(format_spec, fromSI(current_si, current_unit, unit_conversions));
        }
        update_input_text();

        input.disabled = !editable;  // Disable editing if not editable
        // Conditional styling based on editable
        input.className = editable ? 'si-input si-editable' : 'si-input si-readonly';
        input.readOnly = !editable;  // Additional read-only for clarity

        const select = document.createElement('select');
        select.className = 'si-select';
        Object.keys(unit_conversions).forEach(u => {
            const opt = document.createElement('option');
            opt.value = u;
            opt.textContent = u;
            if (u === current_unit) opt.selected = true;
            select.appendChild(opt);
        });

        // Unit change listener (triggers reconversion and syncs all selects)
        select.addEventListener('change', (e) => {
              const new_unit = e.target.value;
            if(new_unit!=current_unit){
                  current_unit=new_unit;
                  update_input_text();
              };
        });

        container.appendChild(input);
        container.appendChild(select);

        el.appendChild(container);
        el.className = 'si-widget';

        // Update on input change (convert to SI; fires on blur/Enter, only if editable)
        input.addEventListener('change', () => {
            if (!editable) return;  // Skip if disabled
            const text = input.value.trim();
            if (!text || text === 'Invalid') return;
            let num;
            try {
                num = parseFloat(text);  // Handles '8.14e3'
                if (isNaN(num)) throw new Error();
            } catch {
                input.value = 'Invalid';
                return;
            }
            current_si = toSI(num, current_unit, unit_conversions);
            model.set('si_value', current_si);
            model.save_changes();
            // Reformat if scientific input
            update_input_text();
//            if (text.toLowerCase().includes('e')) {
//                input.value = formatValue(format_spec, num);
//            }
        });

        // Listen for Python-side value changes (e.g., external updates)
        function si_value_change_handler(){
            current_si = model.get('si_value');
            update_input_text();
        };
        model.on('change:si_value',si_value_change_handler);
        
        function si_editable_change_handler(){
            editable=model.get('editable');
            input.disabled = !editable;  // Disable editing if not editable
            input.className = editable ? 'si-input si-editable' : 'si-input si-readonly';
            input.readOnly = !editable;  // Additional read-only for clarity
        };
        model.on('change:editable',si_editable_change_handler);

        // Initial sync
        model.save_changes();

        return ()=>{
            //clean up event listeners 
            model.off('change:si_value',si_value_change_handler);
            model.off('change:si_editable',si_editable_change_handler);
            //model.save_changes();
        }
    }
    export default { render, initialize };
    """)

    _css = (r"""
    .si-widget {
        display: inline-block;
        margin-top: -3px;
        line-height: 1;
        vertical-align: middle;
    }
    .si-container {
        display: inline-flex;
        gap: 1px;
        align-items: center;
    }
    .si-input {
        width: var(--input-width, 60px);
        padding: var(--input-padding, 1px);
        text-align: right;
        font: inherit;
        box-sizing: border-box;
    }
    .si-input.si-editable {
        border: 1px solid #ccc;
        background-color: #fff;
        cursor: text;
    }
    .si-input.si-readonly {
        border: none;
        background-color: transparent;
        color: #666;
        cursor: default;
    }
    .si-select {
        padding: 0;
        margin: 0;
        text-align: left;
        font: inherit;
    }
    .si-select option {
        padding: 1px 0;
        margin: 0;
        text-align: left;
    }
    """)


    def __init__(self, 
                 unit_conversions: dict, 
                 default_unit=None, 
                 si_value: float = 0.0, 
                 format_spec: str = '.2f',
                 editable: bool = True, 
                 on_change: Optional[Callable[[float], None]] = None,
                 **kwargs):

        # Normalize dict: floats -> [float, 0]
        self.format_spec=format_spec
        normalized_unit_conversions = {}
        for unit, conv in unit_conversions.items():
            normalized_unit_conversions[unit] = conv if isinstance(conv, list) else [conv, 0.0]
        if not default_unit in unit_conversions:
            default_unit = list(unit_conversions.keys())[0]  # First key as default

        # Embed normalized conversions as JSON in JS
        conversions_json = json.dumps(normalized_unit_conversions)
        print(f"{si_value= }")
        super().__init__(
            si_value=si_value,
            defaults={"unit_conversions":normalized_unit_conversions,
                      "default_unit":default_unit,
                      "default_format_spec":format_spec,
                     },
            editable=editable,
            **kwargs
        )
        # Hook on_change to si_value trait (only fires on actual changes)
        if on_change:
            def _on_si_change(change):
                on_change(change['new'])  # Pass new SI value
            self.observe(_on_si_change, names=['si_value'])


@app.class_definition
class SIUnitInput(mo.ui.anywidget):
    def __init__(self,*args,**kwargs):
        super().__init__(SIUnitInputAnywidget(*args,**kwargs))
    def __format__(self, spec: str) -> str:
        import re,json
        base_html = super().__format__(spec)
#        m=re.match(r"(?P<len>\d+)?(?:(?P<grp_sep>[,.])?(?P<prec_sep>[,.])(?P<prec>\d+))?(?P<fmt>[ef])?(?:\[(?P<unit>[^\]]*)\])?",spec)
        spec=spec.strip()
        if spec :
           return f"<span style='--format-spec:{json.dumps(spec)}'>{base_html}</span>"
        else:
           return base_html


@app.cell
def _():
    σ_SB=5.67037442E-8 # [W/(m**2 K)] Stefan-Boltzmann constant
    return (σ_SB,)


@app.cell
def _():
    pre_defined_temperatures=mo.ui.dropdown(
        options={
            "5.18 °C":273.15+5.18,
            "60.0 °F":273.15+(60-32)*5/9,
            "300.0 K":300,
            },
        value=None,
        label="pre-defined values",
        allow_select_none=True)
    return (pre_defined_temperatures,)


@app.cell
def _():
    from itertools import count
    count1=count(1)
    count2=count(1)
    count3=count(1)
    count4=count(1)
    return count1, count2, count3, count4


@app.cell
def _(count1, pre_defined_temperatures):
    T_input = SIUnitInput(
        unit_conversions={"K": 1.0, "°C": [1.0, 273.15], "°F": [5/9, 273.15 - 32 * 5/9]},
        default_unit="°C",
        si_value=pre_defined_temperatures.value or 273.15, 
        editable=pre_defined_temperatures.value==None,  
        format_spec = '.2f'
    )
    print(f"cell1 run count= {next(count1)}")
    return (T_input,)


@app.cell
def _(T_input, count2, pre_defined_temperatures):
    print(f"cell2 run count= {next(count2)}")
    md1=mo.md(f"""
    ## Input Widget with Unit Conversion:
    {pre_defined_temperatures} <-- select <span style="border: 1px solid #999; padding: 0px 10px;margin-right: 05px "> -- </span> for manual input

    T= {T_input:0.2f} = {T_input:0.3f} = {T_input:0.4f} <-- select different units for each widget view

    """)
    md1
    return


@app.cell
def _(T_input, count3, σ_SB):
    print(rf"cell3 run count= {next(count3)}")
    md2=mo.md(rf"""
    <p></p>
    ## Black Body Radiation Example:
    T={T_input:0.0f} SI value: {T_input.si_value:0.2f}K

    $$
     q_\mathrm{{BB}}=\sigma_{{SB}}\cdot T^4 
    ={σ_SB*1e8:0.4f}\times 10^{{-8}}\mathrm{{\frac{{W}}{{m^2\cdot K^4}}}}\cdot({T_input.si_value:0.2f}\,\mathrm K)^4
    ={σ_SB*T_input.si_value**4:.2f}\mathrm{{\frac{{W}}{{m^2 }}}}
     $$
    """
    )
    #print(md2.text)
    md2
    return


@app.cell
def _(T_input, count4):
    print(f"cell4 run count= {next(count4)}")
    T_input.style({"--format-spec":json.dumps("0.3f")})
    return


if __name__ == "__main__":
    app.run()
