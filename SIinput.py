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
    _base_esm = """
    let my_widget_id=0;
function listAllProperties(obj) {{
    const props = new Set();
    let current = obj;
    while (current && current !== Object.prototype) {{
        Object.getOwnPropertyNames(current).forEach(prop => props.add(prop));
        current = Object.getPrototypeOf(current);
    }}
    return Array.from(props).sort();
}}

    function render({{ model, el }}) {{
        let current_si = model.get('si_value') || 0;
        let format_spec = '{FORMAT_SPEC_PLACEHOLDER}' || '.2f';
        let unit_conversions = {CONVERSIONS_PLACEHOLDER};
        let current_unit = el.unit ?? ('{DEFAULT_UNIT_PLACEHOLDER}' || Object.keys(unit_conversions)[0]);
        if(el.unit===undefined){{
        el.unit=current_unit;
        }}
        let editable = model.get('editable') ?? true;
        // Global array to track all selects for cross-instance sync
        if (typeof window.siUnitSelects === 'undefined') {{
            window.siUnitSelects = [];
        }}

        // Create UI elements
        const container = document.createElement('div');
        container.style.display = 'inline-flex';
        container.style.gap = '1px';
        container.style.alignItems = 'center';

        const input = document.createElement('input');
        input.type = 'text';
        input.style.width = '100px';
        input.style.padding = '1px';
        input.style.textAlign = 'right';
        input.placeholder = editable ? 'Enter value' : '';
        input.value = formatValue(format_spec, fromSI(current_si, current_unit, unit_conversions));
        input.disabled = !editable;  // Disable editing if not editable
        // Conditional styling based on editable
        if (editable) {{
            input.style.border = '1px solid #ccc';  // Subtle border for input feel
            input.style.backgroundColor = '#fff';   // White background
            input.style.cursor = 'text';            // Editable cursor
        }} else {{
            input.style.border = 'none';            // No border for text look
            input.style.backgroundColor = 'transparent';  // Transparent bg
            input.style.color = '#666';             // Muted color (adjust to match theme)
            input.style.cursor = 'default';         // Non-interactive cursor
        }}
        input.readOnly = !editable;  // Additional read-only for clarity
        //alert(editable);
        const select = document.createElement('select');
        select.id = 'si-unit-select-' + Date.now() + '-' + Math.random();  // Unique per render
        select.style.padding = '0px 0px';
        select.style.margin = '0px';
        select.style.textAlign = 'left';
        Object.keys(unit_conversions).forEach(u => {{
            const opt = document.createElement('option');
            opt.value = u;
            opt.textContent = u;
            if (u === current_unit) opt.selected = true;
            // Individual option styling: Reduce padding, left-align
            opt.style.paddingLeft = '0px';  // Tight left padding (adjust to '0px' for flush)
            opt.style.padding = '1px 0px';  // Overall: minimal top/bottom/left/right
            opt.style.margin = '0px';
            opt.style.textAlign = 'left';   // Left-align text
            select.appendChild(opt);
        }});
        window.siUnitSelects.push(select);  // Track for sync

        // Unit change listener (triggers reconversion and syncs all selects)
        select.addEventListener('change', (e) => {{
              const new_unit = e.target.value;
              if(new_unit!=current_unit){{
                  current_unit=new_unit;
                  el.unit=current_unit;
                  input.value = formatValue(format_spec, fromSI(current_si, current_unit, unit_conversions));
              }};
        }});

        container.appendChild(input);
        container.appendChild(select);
        el.appendChild(container);
        el.style.display = 'inline-block';
        el.style.marginTop = '-3px';  // Fine-tune upward shift (adjust -1px to -4px if needed)
        el.style.lineHeight = '1';  // Normalize line height
        el.style.verticalAlign = 'middle';  // Aligns with text baseline

        // Dynamic conversions from dict (m * display + b for toSI; inverse for fromSI)
        function toSI(val, unit, conv) {{
            const [m, b = 0] = Array.isArray(conv[unit]) ? conv[unit] : [conv[unit], 0];
            return m * val + b;
        }}

        function fromSI(val, unit, conv) {{
            const [m, b = 0] = Array.isArray(conv[unit]) ? conv[unit] : [conv[unit], 0];
            if (m === 0) return val;  // Avoid div/0
            return (val - b) / m;
        }}
     // Format display value (basic Python-like support)
    function formatValue(spec, val) {{
        try {{
            const fixedMatch = spec.match(/\\.(\\d+)f/);
            const expMatch = spec.match(/\\.(\\d+)e/);
            const precMatch = spec.match(/\\.(\\d+)g/);
            if (expMatch) {{
                const digits = parseInt(expMatch[1]) ?? 2;
                return val.toExponential(digits);
            }} else if (precMatch) {{
                const digits = parseInt(precMatch[1]) ?? 2;
                return val.toPrecision(digits);
            }}
            const digits = parseInt(fixedMatch ? fixedMatch[1] : 2) ?? 2;
            return val.toFixed(digits);
        }} catch {{
            return val.toFixed(2);
        }}
    }}

        // Update on input change (convert to SI; fires on blur/Enter, only if editable)
        input.addEventListener('change', () => {{
            if (!editable) return;  // Skip if disabled
            const text = input.value.trim();
            if (!text || text === 'Invalid') return;
            let num;
            try {{
                num = parseFloat(text);  // Handles '8.14e3'
                if (isNaN(num)) throw new Error();
            }} catch {{
                input.value = 'Invalid';
                return;
            }}
            current_si = toSI(num, current_unit, unit_conversions);
            model.set('si_value', current_si);
            model.save_changes();
            // Reformat if scientific input
            if (text.toLowerCase().includes('e')) {{
                input.value = formatValue(format_spec, num);
            }}
        }});

        // Listen for Python-side value changes (e.g., external updates)
        model.on('change:si_value', () => {{
            current_si = model.get('si_value');
            input.value = formatValue(format_spec, fromSI(current_si, current_unit, unit_conversions));
        }});

        // Initial sync
        model.save_changes();
    }}
    export default {{ render }};
    """

    # Minimal traits (only essential sync)
#    unit = traitlets.Unicode().tag(sync=False)  # Current unit (static)
    si_value = traitlets.Float(0.0).tag(sync=True)  # SI value (reactive)
    editable = traitlets.Bool(True).tag(sync=True)  # Static; (reactive)


    def __init__(self, unit_conversions: dict, default_unit=None, si_value: float = 0.0, format_spec: str = '.2f',
                 editable: bool = True, on_change: Optional[Callable[[float], None]] = None, **kwargs):
        # Normalize dict: floats -> [float, 0]
        self.format_spec=format_spec
        normalized = {}
        for unit, conv in unit_conversions.items():
            normalized[unit] = conv if isinstance(conv, list) else [conv, 0.0]
        if not default_unit in unit_conversions:
            default_unit = list(unit_conversions.keys())[0]  # First key as default

        # Embed normalized conversions as JSON in JS
        conversions_json = json.dumps(normalized)

        esm_with_conversions = self._base_esm.format(
            CONVERSIONS_PLACEHOLDER=conversions_json,
            DEFAULT_UNIT_PLACEHOLDER=default_unit,
            FORMAT_SPEC_PLACEHOLDER=format_spec,  # Inject bare spec (e.g., '.3e')
            )  
        super().__init__(
            unit=default_unit,
            si_value=si_value,
            editable=editable,  # Pass to JS via trait (read-only)
            _esm=esm_with_conversions,
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
        #base_html = super()._repr_html_()
        base_html = super().__format__(spec)
        if spec :
           return f'<span data-format-spec="{spec}" >{base_html}</span>'
        else:
           return base_html


@app.cell
def _():
    # Usage Example (toggle via selector; recreate on change)
    planet_selector = mo.ui.dropdown(options=["Earth", "Mars"], label="Planet")
    temp_conversions = {"K": 1.0, "°C": [1.0, 273.15], "°F": [5/9, 273.15 - 32 * 5/9]}
    planet_temp={
    "Earth":288.15,  # Fixed for Earth
    "Mars":210.0   # Fixed for Mars
    }
    planet_temp_get, planet_temp_set= mo.state(273.15)
    return (
        planet_selector,
        planet_temp,
        planet_temp_get,
        planet_temp_set,
        temp_conversions,
    )


@app.cell
def _(planet_temp_get):
    print("planet temperature changed to:",planet_temp_get())
    return


@app.cell
def _(planet_selector, planet_temp, planet_temp_set):
    if planet_selector.value: 
        planet_temp_set(planet_temp[planet_selector.value])
    print("planet selection changed to",planet_selector.value)
    return


@app.cell
def cellx(planet_selector, planet_temp_get, planet_temp_set, temp_conversions):
    temp_input = SIUnitInput(
        unit_conversions=temp_conversions,
        default_unit="°C",
        si_value=planet_temp_get(),  # From state (fixed or manual)
        on_change=planet_temp_set,   # Updates state if editable
        editable=planet_selector.value is None,  # Manual only if no planet selected
        format_spec = '.2f'
    )
    return (temp_input,)


@app.cell
def _():
    R_input = SIUnitInput(
        unit_conversions={"m":1, "km":1000.0},
        default_unit="km", 
        si_value=20,  # From state (fixed or manual) 
        editable=True,  # Manual only if no planet selected
        format_spec = '.3f'
    )
    return (R_input,)


@app.cell
def _(planet_selector, planet_temp_get, temp_input):
    # In separate cell for reactivity
    ()
    _md=mo.md(f"""

    text {planet_selector} more text, {temp_input:.5f} still more text, **SI Value (K):** {planet_temp_get():0.2f}

    """
    )
    print(_md.text)
    _md
    return


@app.cell
def _(temp_input):
    temp_input
    return


@app.cell
def _():
    #temp_input
    return


@app.cell
def _(R_input):
    R_input
    return


@app.cell
def _():
    T_input = SIUnitInput(
        unit_conversions={"K": 1.0, "°C": [1.0, 273.15], "°F": [5/9, 273.15 - 32 * 5/9]},
        default_unit="°C",
        si_value=273.15+15, 
        editabl=True,  # Manual only if no planet selected
        format_spec = '.2f'
    )
    count1=[]
    count2=[]
    return T_input, count1, count2


@app.cell
def _(T_input, count1):
    count1.append(None)
    print(f"cell1 run count: {len(count1)}")
    md1=mo.md(f"""
    User input:{T_input} SI value: {T_input.si_value:0.2f}K
    """)
    md1
    return (md1,)


@app.cell
def _(T_input, count2):
    count2.append(None)
    print(f"cell2 run count: {len(count2)}")
    md2=mo.md(f"""
    T= {T_input:0.5f+273.15[°C]} = {T_input:0.5f/1.8+255.372[°F]} = {T_input:0.2f[K]}
    """)
    md2
    return (md2,)


@app.cell
def _(md1, md2):
    print(md1.text)
    print(md2.text)
    return


if __name__ == "__main__":
    app.run()
