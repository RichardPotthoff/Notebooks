---
jupyter:
  jupytext:
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

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
[![Run Jupyter Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RichardPotthoff/Notebooks/main?filepath=beta_testing.ipynb)   <- click here to open this file in MyBinder
   
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RichardPotthoff/Notebooks/blob/main/beta_testing.ipynb)   <- click here to open this file in Google Colab
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
# Tests
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## Tests for global "window" instance
<!-- #endregion -->

```js jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
//alert(document.body)
window.test={result:true}
//alert("test.result= "+test.result)
```

```js jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
alert("test.result= "+test.result)
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
| Implementation | Test Result |
|---|---|
| Pythonista Lab v1.0b7 | **Alert** test.result= true |
| Carnets SCI (nb+lab+classic) | **Alert** test.result= true |
| MyBinder (nb+lab) | **Alert** test.result= true |
| Google Colab | *no output* |
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## Test for "IPython", "Jupyter" in JavaScript
<!-- #endregion -->

```js jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
alert(IPython)
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
| Implementation | Test Result |
|:---|:---|
| Pythonista Lab v1.0b7 | *no output* |
| Carnets SCI (nb+lab) | Javascript Error: Can't find variable: IPython |
| Carnets SCI classic | [object Object] |
| MyBinder | Javascript Error: Can't find variable: IPython |
| Google Colab | *no output* |
<!-- #endregion -->

```js jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
alert(Jupyter)
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
| Implementation | Test Result |
|:---|:---|
| Pythonista Lab v1.0b7 | *no output* |
| Carnets SCI (nb+Lab) | Javascript Error: Can't find variable: Jupyter |
| Carnets SCI classic | [object Object] |
| MyBinder (nb+lab)| Javascript Error: Can't find variable: Jupyter |
| Google Colab | *no output* |
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## Test for "element" in Javascript
<!-- #endregion -->

```js jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
alert(element)
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false, "parentCellID": "0d2eab05-5274-4f15-89dd-3a55e1fae3f6"} -->
| Implementation | Test Result |
|:---|:---|
| Pythonista Lab v1.0b7 | *no output* |
| Carnets SCI (nb+Lab) | [object HTMLDivElement] |
| Carnets SCI classic | [object Object] |
| MyBinder (nb+lab)| [object HTMLDivElement] |
| Google Colab | [object HTMLDivElement] |
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## Console output tests
<!-- #endregion -->

```js jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
console.log("My custom message");
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false, "parentCellID": "0d2eab05-5274-4f15-89dd-3a55e1fae3f6"} -->
| Implementation | Test Result |
|:---|:---|
| Pythonista Lab v1.0b7 | *no output* |
| Carnets SCI (nb+Lab) | *no output* |
| Carnets SCI classic | *no output* |
| MyBinder (nb+lab)| *no output* |
| Google Colab | *no output* |
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false, "parentCellID": "4b09b153-9fd5-4c1a-999e-7f8e5eca672b"} -->
## Console hook tests
<!-- #endregion -->

```js jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false}
try{console._o=console._o||{log:console.log,error:console.error,warn:console.warn,info:console.info};[['log'],['error','red'],['warn'],['info']].forEach(([t,c="black"])=>{console[t]=console._o[t];let d=element.consoleOutput;if(!d){d=document.createElement("div");d.id="console-output";d.style.cssText="white-space:pre-wrap;font-family:monospace;padding:0px;background:#f0f0f0;line-height:1.1;";element.appendChild(d);element.consoleOutput=d;}let o=console[t],n=o;while(n&&n.toString().indexOf("[native code]")<0)n=n.apply?function(...a){return n.apply(this,a);}:null;o=function(...a){(n||console._o[t]).apply(console,a);let s=a.map(e=>typeof e==='object'?JSON.stringify(e,null,2):e).join(' ');const m=document.createElement("div");m.innerHTML=`[${t.toUpperCase()}] ${s}`;m.style.cssText=`margin:0;line-height:1.1;padding:0px 0;color:${c};`;d.appendChild(m);};console[t]=o;});

console.log("My custom message");

}catch(e){console.error("Error:",e);}finally{[['log'],['error','red'],['warn'],['info']].forEach(([t])=>console._o&&console._o[t]&&(console[t]=console._o[t]));}
```

<!-- #region jupyter={"outputs_hidden": false} pythonista={"disabled": false} -->
| Implementation | Test Result |
|:---|:---|
| Pythonista Lab v1.0b7 | *no output* |
| Carnets SCI (nb+Lab) | [LOG] My custom message |
| Carnets SCI classic | *no output* |
| MyBinder (nb+lab)| [LOG] My custom message |
| Google Colab | [LOG] My custom message |
<!-- #endregion -->

## Data access between Javascript and Python


### ipywidgets

```python
import ipywidgets as widgets
from IPython.display import display, Javascript

# Create a widget to hold Python data
output = widgets.HTML(value="<div id='python-data'>[1, 2, 3]</div>")
display(output)

# JavaScript to read the data
display(Javascript(
"""
var data = document.getElementById('python-data').innerText;
alert("Python data: " + data);
"""))
```

<!-- #region jupyter={"outputs_hidden": false} pythonista={"disabled": false} -->
| Implementation | Test Result |
|:---|:---|
| Pythonista Lab v1.0b7 |  |
| Carnets SCI (nb+Lab) | Javascript Error: null is not an object |
| Carnets SCI classic |  |
| MyBinder (nb+lab)|  |
| Google Colab |  |
<!-- #endregion -->

```js
var data = document.getElementById('python-data').innerText;
alert("Python data: " + data);
```

<!-- #region jupyter={"outputs_hidden": false} pythonista={"disabled": false} -->
| Implementation | Test Result |
|:---|:---|
| Pythonista Lab v1.0b7 |  |
| Carnets SCI (nb+Lab) | Python Data [1,2,3] |
| Carnets SCI classic | *no output* |
| MyBinder (nb+lab)|  |
| Google Colab |  |
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## Print pdf
<!-- #endregion -->

```python jupyter={"source_hidden": true} pythonista={"disabled": false}
#dodecahedron pattern 
from math import tan,sin
import numpy as np
from matplotlib import pyplot as plt
from cmath import pi,acos,exp,sqrt

def plotArc(ax,P0,n0,l,da,*args,tol=0.001,**kwargs):
  if l==0:
    return
  x=np.linspace(0,l,max(2,int(abs(6*(da/(2*pi)))),int(l//(2*abs(2*l/da*tol)**0.5)+1))if (da!=0) and (l!=0) else 2)
  phi2=x/l*da/2
  p=P0+x*np.sinc(phi2/pi)*n0*np.exp(1j*phi2)
  ax.plot(p.real,p.imag,*args,**kwargs)
    
def plotArcchain(ax,P0,n0,arcs,*args,**kwargs):
    p=P0
    n=n0
    for l,da in arcs:
        plotArc(ax,p,n,l,da,*args,**kwargs)
        p+=l*np.sinc(da/(2*pi))*n*exp(1j*da/2)
        n*=exp(1j*da)

deg=pi/180.0

pentagon=([(1.0,0),(0,72*deg)]*2+[(1,0),(0,-36*deg)]*2+[(1.0,0),(0,72*deg)])
scoreline=[(0.0,-108*deg)]+((pentagon*5)[2:-1]+[(0.0,-36*deg)] +(pentagon*5)[2:])+[(1.0,0),(0.0,180*deg),(1.5,0)]

fs=0.5#flap start
fsa=80*deg#start angle
fsb=20*deg#flap start edge curvature 80deg+20deg=100deg at root -> 'snap'-fit
fe=0.95#flap end
fea=30*deg#end angle
fw=0.16#flap width
flap=[(fs,0),(0.,-fsa-fsb),(fw/sin(fsa)*(fsb/sin(fsb) if fsb!=0.0 else 1.0),2*fsb),(0.0,fsa-fsb),
       (fe-fs-fw*(1/tan(fsa)+1/tan(fea)),0),(0,fea),(fw/sin(fea),0),(0,-fea),(1.0-fe,0)]
flapped_pentagon=[*flap,(0,72*deg)]*2+[*flap,(0,-144*deg)]+[*flap,(0.0,72*deg)]
cutline=[(0.0,-108*deg)]+(flapped_pentagon*5)[len(flap)+1:-1]+[(0.0,-36*deg)]+(flapped_pentagon*5)[len(flap)+1:-1]+[(0,-108*deg),(0.25,0)]


from matplotlib import pyplot as plt 
plt.close()
#DIN_A=4
#plt.figure(figsize=(1000*2**(-DIN_A/2+1/4)/25.4,1000*2**(-DIN_A/2-1/4)/25.4))
plt.figure(figsize=(11,8.5))
plotArcchain(plt.gca(),0.0 + 0.0j, 0.0 +1.0j,scoreline,'k:',)
plotArcchain(plt.gca(),0.0 + 0.0j, 0.0 +1.0j,cutline,'k-')
plt.plot(0,0,'k-',label='cut')
plt.plot(0,0,'k:',label='fold up')
plt.gca().set_aspect('equal')
xlim=plt.gca().get_xlim()
ylim=plt.gca().get_ylim()
plt.legend()
plt.gca().set_axis_off()
plt.savefig('dodecahedron.pdf')
plt.show()
(xlim[1]-xlim[0])/(ylim[1]-ylim[0])
```

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
| Implementation | Test Result |
|:---|:---|
| Pythonista Lab v1.0b7 | ModuleNotFoundError: No module named 'fontTools', (no pdf file, but plot in notebook O.K.) |
| Carnets SCI (nb+Lab+classic) | Notebooks/dodecahedron.pdf (10kB) |
| MyBinder (nb+lab)| /dodecahedron.pdf (9.75kB) |
| Google Colab | content/dodekahedron.pdf (9.75kB)  |
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} pythonista={"disabled": false} -->
# Appendix
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
## Expanded console hook function
<!-- #endregion -->

```js jupyter={"source_hidden": true} pythonista={"disabled": false}
try {
    // Store original system console methods
    console._o = console._o || {
        log: console.log,
        error: console.error,
        warn: console.warn,
        info: console.info
    };

    [['log'], ['error', 'red'], ['warn'], ['info']].forEach(([method, color = "black"]) => {
        // Reset to system console to clear prior hooks
        console[method] = console._o[method];

        // Create or reuse console-output div
        let consoleOutput = element.consoleOutput;
        if (!consoleOutput) {
            consoleOutput = document.createElement("div");
            consoleOutput.id = "console-output";
            consoleOutput.style.cssText = "white-space: pre-wrap; font-family: monospace; padding: 0px; background: #f0f0f0; line-height: 1.1;";
            element.appendChild(consoleOutput);
            element.consoleOutput = consoleOutput;
        }

        // Find native console by traversing chain
        let original = console[method], native = original;
        while (native && native.toString().indexOf("[native code]") < 0) {
            native = native.apply ? function(...args) { return native.apply(this, args); } : null;
        }

        // Define new hook
        original = function(...args) {
            (native || console._o[method]).apply(console, args); // Forward to native or original
            let message = args.map(arg => {
                return typeof arg === 'object' ? JSON.stringify(arg, null, 2) : arg;
            }).join(' ');
            const messageDiv = document.createElement("div");
            messageDiv.innerHTML = `[${method.toUpperCase()}] ${message}`;
            messageDiv.style.cssText = `margin: 0; line-height: 1.1; padding: 0px 0; color: ${color};`;
            consoleOutput.appendChild(messageDiv);
        };
        console[method] = original;
    });

    // Cell-specific code
    console.log("Test message 1");
    console.log({ key: "value" });
    console.error("Test error");
    console.warn("Test warning");
    console.info("Test info");
} catch (e) {
    console.error("Error:", e);
} finally {
    // Restore system console
    [['log'], ['error', 'red'], ['warn'], ['info']].forEach(([method]) => {
        if (console._o && console._o[method]) {
            console[method] = console._o[method];
        }
    });
}
```

## wrapper fumction to capture console output fron JavaScript code

```python
def capture_console(code):
    return (
"""
try{console._o=console._o||{log:console.log,error:console.error,warn:console.warn,info:console.info};[['log'],['error','red'],['warn'],['info']].forEach(([t,c="black"])=>{console[t]=console._o[t];let d=element.consoleOutput;if(!d){d=document.createElement("div");d.id="console-output";d.style.cssText="white-space:pre-wrap;font-family:monospace;padding:0px;background:#f0f0f0;line-height:1.1;";element.appendChild(d);element.consoleOutput=d;}let o=console[t],n=o;while(n&&n.toString().indexOf("[native code]")<0)n=n.apply?function(...a){return n.apply(this,a);}:null;o=function(...a){(n||console._o[t]).apply(console,a);let s=a.map(e=>typeof e==='object'?JSON.stringify(e,null,2):e).join(' ');const m=document.createElement("div");m.innerHTML=`[${t.toUpperCase()}] ${s}`;m.style.cssText=`margin:0;line-height:1.1;padding:0px 0;color:${c};`;d.appendChild(m);};console[t]=o;});
"""  +
code + 
"""
}catch(e){console.error("Error:",e);}finally{[['log'],['error','red'],['warn'],['info']].forEach(([t])=>console._o&&console._o[t]&&(console[t]=console._o[t]));}
"""
)
```

##  wrappers around ipywidget.Text input widgets for sending data from js to Python

```python
import ipywidgets as widgets
from ipywidgets import Output
from IPython.display import display, Javascript, clear_output

# Clear previous outputs to remove old widgets

clear_output(wait=True)

# Define NamedText class
class NamedText(widgets.Text):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
        try:
            self.css_classes = [f'python_{name}']  # For ipywidgets 8.x
            print(f"Set css_classes for {name}: {self.css_classes}")
        except AttributeError:
            self.dom_classes = [f'python_{name}']  # Fallback for ipywidgets 7.x
            print(f"Set dom_classes for {name}: {self.dom_classes}")
        self.add_class(f'python_{name}')
    
    @property
    def name(self):
        return self._name

# Create widgets
name = NamedText(name='name', value='First', description='Name:')
email = NamedText(name='email', value='Second', description='Email:')

# Hide the widgets
name.layout.display = 'none'
email.layout.display = 'none'

# Display the widgets
display(name, email)
output1=Output()
display(output1)
# Add change handler  
def on_name_change(change):
    with output1:
      print(f"Name changed to: {change['new']}")

name.observe(on_name_change, names='value')
```

```python
# Query the DOM
def query_dom():
    js_code = """
    const elements = document.querySelectorAll('.widget-text');
    const results = [];
    elements.forEach((el, index) => {
        const input = el.querySelector('input');
        const classes = Array.from(el.classList);
        const inputValue = input ? input.value : 'No input found';
        results.push({
            index: index,
            classes: classes,
            inputValue: inputValue,
            style: el.style.display || 'Not set'
        });
    });
    console.log('DOM Query Results:', JSON.stringify(results));
    """
    display(Javascript(capture_console(js_code)))

query_dom()
```

```javascript
try{console._o=console._o||{log:console.log,error:console.error,warn:console.warn,info:console.info};[['log'],['error','red'],['warn'],['info']].forEach(([t,c="black"])=>{console[t]=console._o[t];let d=element.consoleOutput;if(!d){d=document.createElement("div");d.id="console-output";d.style.cssText="white-space:pre-wrap;font-family:monospace;padding:0px;background:#f0f0f0;line-height:1.1;";element.appendChild(d);element.consoleOutput=d;}let o=console[t],n=o;while(n&&n.toString().indexOf("[native code]")<0)n=n.apply?function(...a){return n.apply(this,a);}:null;o=function(...a){(n||console._o[t]).apply(console,a);let s=a.map(e=>typeof e==='object'?JSON.stringify(e,null,2):e).join(' ');const m=document.createElement("div");m.innerHTML=`[${t.toUpperCase()}] ${s}`;m.style.cssText=`margin:0;line-height:1.1;padding:0px 0;color:${c};`;d.appendChild(m);};console[t]=o;});

// Wrapper function to send data to a widget's input element
function sendData(classname, data) {
    // Find the input element by classname
    const input = document.querySelector(`.widget-text.${classname} input`);
    if (!input) {
        console.error(`Input not found for class: ${classname}`);
        return false;
    }
    
    // Stringify data if it's an object
    const value = typeof data === 'object' ? JSON.stringify(data) : data.toString();
    
    // Set the input value
    input.value = value;
    
    // Trigger a change event to sync with Python
    const event = new Event('change', { bubbles: true });
    input.dispatchEvent(event);
    
    console.log(`Sent data to ${classname}:`, value);
    return true;
}

// Wrapper function to get data from a widget's input element
function getData(classname) {
    // Find the input element by classname
    const input = document.querySelector(`.widget-text.${classname} input`);
    if (!input) {
        console.error(`Input not found for class: ${classname}`);
        return null;
    }
    
    console.log(`Got data from ${classname}:`, input.value);
    return input.value;
}

// Ensure functions are globally available (optional, for reuse in other cells)
window.sendData = sendData;
window.getData = getData;

}catch(e){console.error("Error:",e);}finally{[['log'],['error','red'],['warn'],['info']].forEach(([t])=>console._o&&console._o[t]&&(console[t]=console._o[t]));}
```

```javascript
try{console._o=console._o||{log:console.log,error:console.error,warn:console.warn,info:console.info};[['log'],['error','red'],['warn'],['info']].forEach(([t,c="black"])=>{console[t]=console._o[t];let d=element.consoleOutput;if(!d){d=document.createElement("div");d.id="console-output";d.style.cssText="white-space:pre-wrap;font-family:monospace;padding:0px;background:#f0f0f0;line-height:1.1;";element.appendChild(d);element.consoleOutput=d;}let o=console[t],n=o;while(n&&n.toString().indexOf("[native code]")<0)n=n.apply?function(...a){return n.apply(this,a);}:null;o=function(...a){(n||console._o[t]).apply(console,a);let s=a.map(e=>typeof e==='object'?JSON.stringify(e,null,2):e).join(' ');const m=document.createElement("div");m.innerHTML=`[${t.toUpperCase()}] ${s}`;m.style.cssText=`margin:0;line-height:1.1;padding:0px 0;color:${c};`;d.appendChild(m);};console[t]=o;});

sendData('python_name', { first: 'John', last: 'Doe' });
const emailValue = getData('python_email');

console.log("EmailValue:",emailValue);

}catch(e){console.error("Error:",e);}finally{[['log'],['error','red'],['warn'],['info']].forEach(([t])=>console._o&&console._o[t]&&(console[t]=console._o[t]));}
```

```python
print(name.value)   # Should print '{"first":"John","last":"Doe"}'
print(email.value)  # Should print 'Second'

# Query the DOM again to confirm only two widgets exist
query_dom()
```

### find output div in DOM

```python
import uuid
from IPython.display import display, Javascript, HTML

# Generate a unique ID for this output
unique_id = str(uuid.uuid4())

# Display a marker div with the unique ID
display(HTML(f'<div id="output-marker-{unique_id}"></div>'))

# JavaScript to append to the marker using MutationObserver with timeout and optional debugging
def wrap_code(js_code, output_id, timeout_ms=100, alerts=False, logging=False):
    return f"""
    (async function(){{
        function waitForElement(selector) {{
            return new Promise((resolve, reject) => {{
                const element = document.querySelector(selector);
                if (element) {{
                    resolve(element);
                    return;
                }}
                const observer = new MutationObserver(() => {{
                    const targetElement = document.querySelector(selector);
                    if (targetElement) {{
                        observer.disconnect();
                        resolve(targetElement);
                    }}
                }});
                observer.observe(document.body, {{ childList: true, subtree: true }});
            }});
        }}
        function timeout(ms) {{
            return new Promise((resolve, reject) => {{
                setTimeout(() => {{
                    reject(new Error('Timeout after ' + ms + 'ms'));
                }}, ms);
            }});
        }}
        try {{
            const element = await Promise.race([
                waitForElement('#output-marker-{output_id}'),
                timeout({timeout_ms})
            ]);
            if (!element) {{
                {'console.error("output-marker-' + output_id + ' not found");' if logging else ''}
                {'alert("output-marker-' + output_id + ' not found");' if alerts else ''}
                return;
            }}
            {js_code}
        }} catch (error) {{
            {'console.error("Error: " + error.message);' if logging else ''}
            {'alert("Error: " + error.message);' if alerts else ''}
        }}
    }})();
    """

my_code = """
var newElement = document.createElement('div');
newElement.innerHTML = '<p style="color: purple; font-weight: bold;">Appended to THIS cell\\'s output!</p>';
element.appendChild(newElement);
"""

# Execute the wrapped code
wrapped_code = wrap_code(my_code, unique_id, timeout_ms=100)
#print(wrapped_code)
# try it with a nonexisting unique_id to test the error messages
#wrapped_code = wrap_code(my_code, unique_id[:-5], timeout_ms=100, alerts=True, logging=True)
#print(wrapped_code)
display(Javascript(wrapped_code))
```

## test results template

<!-- #region jupyter={"outputs_hidden": false, "source_hidden": false} pythonista={"disabled": false} -->
| Implementation | Test Result |
|:---|:---|
| Pythonista Lab v1.0b7 | |
| Carnets SCI (nb+Lab+classic) | |
| MyBinder (nb+lab)| |
| Google Colab |  |
<!-- #endregion -->

```python
from ipykernel.comm import comm
```

```python

```

```python

```
