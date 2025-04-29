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

# Implementation of a virtual file system 

```python
import io
import builtins
import sys
import importlib.util
import textwrap

def find_file(data, path_parts, mode=None):
    """
    Recursively find a file in the nested dictionary.
    
    Args:
        data (dict): The current dictionary level.
        path_parts (list): List of path parts (e.g., ['mydir', '__init__.py']).
        mode (str, optional): File mode for creating entries ('w', 'a', etc.).
    
    Returns:
        tuple: (folder, filename, content) if found.
    
    Raises:
        FileNotFoundError: If the path doesn’t exist and mode doesn’t allow creation.
    """
    if not path_parts:
        raise FileNotFoundError("Invalid path: empty path parts")
    key = path_parts[0]
    if len(path_parts) == 1:  # Leaf node (file)
        if key not in data or isinstance(data[key], dict):
            if mode and ('w' in mode or 'a' in mode):
                data[key] = ''
            else:
                raise FileNotFoundError(f"No such file: {'/'.join(path_parts)}")
        return data, key, data[key]
    else:  # Directory
        if key not in data or not isinstance(data[key], dict):
            if mode and ('w' in mode or 'a' in mode):
                data[key] = {}
            else:
                raise FileNotFoundError(f"No such directory: {'/'.join(path_parts)}")
        return find_file(data[key], path_parts[1:], mode)

def create_vfs_open(vfs_data, mount_point='/vfs'):
    """
    Create a simple Open function that returns an io.StringIO object
    for files in the vfs_data nested dictionary. Supports text mode only.
    
    Args:
        vfs_data (dict): Nested dictionary mapping paths to content.
        mount_point (str): Virtual mount point (e.g., '/vfs').
    
    Returns:
        function: An Open function that mimics builtins.open for the virtual filesystem.
    """
    def vfs_open(path, mode='r', *args, **kwargs):
        if 'b' in mode:
            raise ValueError("Binary mode ('b') is not supported; use text mode instead")

        if not path.startswith(mount_point + '/'):
            raise FileNotFoundError(f"Path {path} not in virtual filesystem {mount_point}")
        vfs_path = path[len(mount_point):].lstrip('/')
        if not vfs_path:
            raise FileNotFoundError(f"No such file: {path}")

        parts = vfs_path.split('/')
        folder, filename, content = find_file(vfs_data, parts, mode)

        buffer = io.StringIO(content)

        if 'w' in mode or 'a' in mode or '+' in mode:
            original_close = buffer.close
            def new_close():
                folder[filename] = buffer.getvalue()
                original_close()
            buffer.close = new_close

        if 'a' in mode:
            buffer.seek(0, io.SEEK_END)
        elif 'w' in mode:
            buffer.seek(0)
            buffer.truncate()

        return buffer

    return vfs_open

def load_module(module_name, vfs_path, vfs_data):
    """
    Manually load a module from the vfs_data dictionary.
    """
    module_parts = module_name.split('.')
    vfs_parts = vfs_path.strip('/').split('/')

    current_path = []
    current_name = []
    for i, (part, mod_part) in enumerate(zip(vfs_parts[:-1], module_parts[:-1])):
        current_path.append(part)
        current_name.append(mod_part)
        init_path = '/'.join(current_path) + '/__init__.py'
        package_name = '.'.join(current_name)
        
        if package_name not in sys.modules:
            init_parts = init_path.split('/')
            try:
                _, _, code = find_file(vfs_data, init_parts)
            except FileNotFoundError:
                raise ValueError(f"No __init__.py found at: {init_path}")
            
            spec = importlib.util.spec_from_loader(package_name, None)
            package = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = package
            package.__path__ = ['/vfs/' + '/'.join(current_path)]
            exec(code, package.__dict__)

    try:
        _, _, code = find_file(vfs_data, vfs_parts)
    except FileNotFoundError:
        raise ValueError(f"No module found at: {vfs_path}")
    
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_loader(module_name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    if vfs_path.endswith('__init__.py'):
        module_path = '/'.join(vfs_parts[:-1])
        module.__path__ = ['/vfs/' + module_path]

    exec(code, module.__dict__)
    return module

# Example program that reads a config file
def read_config(config_path, open_func=builtins.open):
    with open_func(config_path) as f:
        return f.read()

# Virtual filesystem data (nested dictionary)
# Use raw triple-quoted strings with textwrap.dedent to preserve exact content
vfs_data = {
    'mydir': {
        '__init__.py': textwrap.dedent(r"""
            # Empty __init__.py
        """).strip(),
        'myfile.py': textwrap.dedent(r"""
            comment_pattern = r'//.*?(?:\n|$)'  # include the trailing newline
            def example():
                '''This is a docstring with "quotes" inside'''
                print("Hello from myfile!")
            example()
        """).strip(),
        'subdir': {
            '__init__.py': textwrap.dedent(r"""
                # Empty __init__.py
            """).strip(),
            'anotherfile.py': textwrap.dedent(r"""
                def foo():
                    '''This is another docstring'''
                    print("Foo!")
            """).strip()
        }
    },
    'config.ini': textwrap.dedent(r"""
        [settings]
        value=42
    """).strip()
}

# Create the custom Open function
vfs_open = create_vfs_open(vfs_data, mount_point='/vfs')

# Load modules
load_module('mydir', '/mydir/__init__.py', vfs_data)
load_module('mydir.myfile', '/mydir/myfile.py', vfs_data)
load_module('mydir.subdir', '/mydir/subdir/__init__.py', vfs_data)
load_module('mydir.subdir.anotherfile', '/mydir/subdir/anotherfile.py', vfs_data)

# Use the modules
import mydir.myfile  # Outputs: Hello from myfile!
from mydir.subdir import anotherfile
anotherfile.foo()  # Outputs: Foo!

# Use the program with the virtual filesystem
config_content = read_config('/vfs/config.ini', open_func=vfs_open)
print(config_content)  # Outputs: [settings]\nvalue=42

# Test writing to the virtual filesystem
with vfs_open('/vfs/config.ini', 'w') as f:
    f.write('[settings]\nvalue=100')
with vfs_open('/vfs/config.ini') as f:
    print(f.read())  # Outputs: [settings]\nvalue=100

# Test binary mode (should raise an error)
try:
    with vfs_open('/vfs/config.ini', 'rb') as f:
        print(f.read())
except ValueError as e:
    print(e)  # Outputs: Binary mode ('b') is not supported; use text mode instead

# Imports still work because modules are cached
import mydir.myfile  # Outputs nothing, uses cached module
```

```python
vfs['es6_html_to_iife_html.py']=r"""
import re
import os
from bs4 import BeautifulSoup
from collections import defaultdict
import re

def combine_patterns(*patterns):
  combined_pattern ='|'.join(f'(?P<pattern{i}>'+pattern[0]+')' for i,pattern in enumerate(patterns))
  return (re.compile(combined_pattern,flags=re.MULTILINE),patterns)
  
def combined_re_sub(content,combined_patterns):
  compiled_re,patterns=combined_patterns
  def callback(match):
    for key,group in match.groupdict().items():
      if group and key.startswith('pattern'):
        i=int(key[7:])
        return patterns[i][1](match)
  return compiled_re.sub(callback,content)
  
#regexes for common javascript patterns:
string_pattern = r"'(?:[^'\\]|\\.)*'|" + r'"(?:[^"\\]|\\.)*"|'
multiline_string_pattern = r'`(?:[^`\\]|\\.)*`'
comment_pattern = r'//.*?(?:\n|$)'#include the trailing newline
multiline_comment_pattern = r'/\*[\s\S]*?\*/'
delimiters=r'[=({:<>;,?%&|*+-/' #removing ]}) from delimiters because of problems with asi not inserting semicolons if there is a \n behind the delimiter
whitespaces_to_right_of_delimiter =r'(?<=['+delimiters+r'])\s*'
whitespaces_to_left_of_delimiter =r'\s*(?=['+delimiters+'\]})'+r'])'
whitespaces_containing_newline=r'\s*\n\s*'
two_or_more_whitespaces = r'\s\s+'
  
combined_minify_patterns=combine_patterns(
    (string_pattern, lambda match:match.group()),           #detect strings, and put them back unminified
    (multiline_string_pattern, lambda match:match.group()), #detect strings, and put them back unminified
    (multiline_comment_pattern, lambda match:''),           #remove all comments 
    (comment_pattern, lambda match:''),                     #remove all comments
    (whitespaces_to_right_of_delimiter,lambda match:''),    #delete whitespaces if there is a delimiter to the left
    (whitespaces_to_left_of_delimiter,lambda match:''),     #delete whitespaces if there is a delimiter to the right
    (whitespaces_containing_newline,lambda match:'\n'),     #replace newline+whitespaces with a single newline
    (two_or_more_whitespaces,lambda match:' '),             #replace span of >=2 whitspaces with single whitespace
    )

minify_javascript=lambda code:combined_re_sub(code,combined_minify_patterns)      

def add_exports(exportlist,exports):
  for item in exportlist.split(','):
    name,*alias=item.split('as')
    alias=alias[0] if alias else name
    exports[alias.strip()]=name.strip()
  return ''

def convert_es6_to_iife(content, module_filename=None, minify=False):
  imports={}
  import_pattern = r'(?=^|;)\s*(import\s+(?:(?:(?:(?P<default_import>\w+)(?:[,]|\s)\s*)?(?:(?P<import_group>\{[^}]*\}\s)|(?:\*\s+as\s+(?P<module_alias>\w+))\s)?)\s*from\s+)?[\'"](?P<module_path>[^"\']+)[\'"]\s*;?)'
  
  def import_callback(match):
      groupdict=match.groupdict()
      default_import=groupdict['default_import'] # these are the named groups in the regular expression
      import_group=groupdict['import_group']
      module_alias=groupdict['module_alias']
      module_path=groupdict['module_path'].strip()
      module_filename=os.path.basename(module_path)
      imports[module_filename]=module_path
      result=[]
      if import_group:
        import_group=re.sub(r'(\w+)\s*as\s*(\w+)',r'\1 : \2',import_group.strip()) #replace 'as' with ':'
        result.append(f'let {import_group.strip()} = modules["{module_filename}"];')
      if module_alias:result.append(f'let {module_alias.strip()} = modules["{module_filename}"];')
      if default_import:result.append(f'let {default_import.strip()} = modules["{module_filename}"].default;')
      return '\n'.join(result)
      
  exports={}
  export_pattern = r'(?=^|;)\s*(export\s+(?P<export_default>default\s+)?(?:(?P<export_type>function|const|let|var|class)\s+)?(?P<export_name>\w+)\s*)'
  
  def export_callback(match):
      groupdict=match.groupdict()
      export_type=groupdict['export_type']
      export_name=groupdict['export_name'].strip()
      exports[export_name]=export_name # possibly add alias syntax later
      if groupdict['export_default']:
        exports['default']=export_name;
      if export_type:
        return export_type+' '+export_name #remove the 'export' and 'default' keywords
      else:
        return ''
      
  # here we are parsing for import and export patterns.
  # strings and comment patterns are detected simultaneously, thus preventing the detection of 
  # import/export patterns inside of strings and comments
  combined_es6_to_iife_patterns=combine_patterns(
      (string_pattern, lambda match:match.group()), #detect strings, and put them back unchanged
      (multiline_string_pattern, lambda match:match.group()),    #       
      (comment_pattern, (lambda match:'') if minify else (lambda match:match.group())), #remove comments only if minify
      (multiline_comment_pattern, (lambda match:'') if minify else (lambda match:match.group())), #
      (import_pattern,import_callback),#parse import statements, and replace them with equivalent let statements
      (export_pattern,export_callback),#parse export statements, collect export names, remove 'export [default]'
      (r'(?=^|;)\s*(export\s+\{(?P<export_list>[^}]*)\}\s*;?)', lambda match:add_exports(match.group('export_list'), exports) ), # ad-hoc pattern for grouped exports: " export {f1, f2 as g, ...}; "
      )
  
  #the next line does all the work: the souce code is modified by the callback functions, and the
  #filenames and pathnames of the imported modules the and names of the exported symbols are collected 
  #in the 'imports' and 'exports' dictionaries. 
  content=combined_re_sub(content,combined_es6_to_iife_patterns)
  
  if exports:  # Only add the export object if there are exports
      iife_wrapper = f'\n(function(global) {{\n{content}\nif(!("modules" in global)){{\n global["modules"]={{}}\n}}\nglobal.modules["{module_filename}"] = {{{",".join((str(key)+":"+str(value) if value and (key!=value) else str(key)) for key,value in exports.items())}}} ;\n}})(window);'
  else:
      iife_wrapper = f'\n(function(global) {{\n{content}\n}})(window);'
      
  if minify:
      iife_wrapper = minify_javascript(iife_wrapper)
  
  return iife_wrapper,imports

def gather_dependencies(content, processed_modules, dependencies, in_process=None, module_dir=None, module_filename=None, minify=False):
    if in_process==None:
      in_process=set()
    if module_filename:
      if module_filename in processed_modules:
        if module_filename in in_process:
          print(f'Circular dependency detected: Module "{module_filename}" is already being processed.')
        return ""
      else:
        in_process.add(module_filename)
        processed_modules.add(module_filename)

    # Process dependencies first
    print(f'Processing module "{module_filename if module_filename else "html <script>"}"')
        # Convert the module itself 
    converted,imports = convert_es6_to_iife(content, module_filename, minify=minify)
    dependency_content = ""
    for ifile_name,ifile_path in imports.items():
        dependencies[module_filename].add(ifile_name)
        full_path = os.path.join(os.path.dirname(module_dir), ifile_path)
        imodule_dir=os.path.dirname(full_path)
        with open(full_path, 'r') as f:
           content = f.read()
        dependency_content += gather_dependencies(content, processed_modules, dependencies,in_process,module_dir=imodule_dir,module_filename=ifile_name, minify=minify)
    if module_filename:
      in_process.remove(module_filename)
    return dependency_content + converted

def process_html(html_path,minify=False,output_file='output.html'):
    with open(html_path, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    processed_modules = set()
    dependencies = defaultdict(set)
    for style in soup.find_all('style'):
      style.string=minify_javascript(style.string)
    for script in soup.find_all('script'):
        if script.get('type') == 'module':
            module_path = script.get('src',None)
            if module_path!=None:
                full_path = os.path.join(os.path.dirname(html_path), module_path)
                module_dir = os.path.dirname(full_path)
                module_filename = os.path.basename(full_path)
                # Gather all dependencies for this module
                with open(full_path, 'r') as f:
                    content = f.read()
                del script['src']  # Remove the src attribute as we've included the content
            else:
                content=script.string
                #module_filename=None
                module_filename=script.get('name')
                module_dir=os.path.dirname(html_path)
            script['type'] = 'text/javascript'  # Change type to standard JavaScript
            # Insert the converted IIFE content for this module and its dependencies
            iife_content = gather_dependencies(content, processed_modules, dependencies,  
                module_dir=module_dir, module_filename=module_filename,  minify=minify)
            script.string = iife_content
        else:
            # For regular scripts, insert their content
            script_path = script.get('src',None)
            if script_path:
               with open(os.path.join(os.path.dirname(html_path), script['src']), 'r') as f:
                   if minify:
                     script.string = minify_javascript(f.read())
                   else:
                     script.string = f.read()
               del script['src']
            else:
                if minify:
                   script.string=minify_javascript(script.string)

    with open(output_file, 'w') as file:
        file.write(str(soup))
  
if __name__ == "__main__":
#    module_filename='index.js'
#    print(convert_es6_to_iife(open(module_filename).read(),module_filename=module_filename,minify=False)[0])
#    raise Exception
    from time import perf_counter
    t1=perf_counter()
    print(os.getcwd())
    
    html_file = "modular.html"
    process_html(html_file,minify=True,output_file='index.html')
    print("HTML processing completed with modules converted to IIFE.")
    t2=perf_counter()
    print(f'{t2-t1=}')
    
'''    
    os.chdir("/private/var/mobile/Containers/Data/Application/77881549-3FA6-4E4B-803F-D53B172FC865/Documents/www")
    html_file = "webgl-3d-camera-look-at-heads.html"
    process_html(html_file,minify=True)
    print("HTML processing completed with modules converted to IIFE.")
'''
    
"""
```

# Python generating JavaScript code for m4.js

```python
import sympy as sp
import os,sys
theta,phi,d,dx,dy,dz,x_t,y_t,z_t=sp.symbols('theta,phi,d,dx,dy,dz,x_t,y_t,z_t')
Matrix=sp.Matrix
sin=sp.sin
cos=sp.cos
tan=sp.tan
pi=sp.pi
pi_2=sp.pi/2
deg=sp.pi/180

def tLat(v):
  dx,dy,dz,*_=v
  return sp.Matrix(sp.BlockMatrix([
    [sp.eye(3),     sp.Matrix([dx,dy,dz])],
    [sp.zeros(1,3), sp.ones(1,1)]
    ]))
  
def xRot(alpha):
  return sp.diag(sp.rot_axis1(-alpha),1)

def yRot(alpha):
  return sp.diag(sp.rot_axis2(-alpha),1)

def zRot(alpha):
  return sp.diag(sp.rot_axis3(-alpha),1)

def scal(v):
  sx,sy,sz,*_=v
  return sp.diag(sx,sy,sz,1)

def persp(fov, aspR, near, far):
    f = tan(pi/2 - fov/2);
    nfInv = 1 / (near - far);
    return Matrix([
      [f / aspR, 0, 0, 0],
      [0, f, 0, 0],
      [0, 0, (near + far) * nfInv, near * far * nfInv * 2],
      [0, 0, -1, 0]
    ])
    
def rot_vector(V,theta=None):
  absV=abs(V)
  if theta==None:
    theta=absV
  x,y,z,*_=V#/absV
  c=cos(theta)
  s=sin(theta)
  v=Matrix([x,y,z])
  return (
    (1-c)*  v@(v.T) + 
      c  * sp.eye(3)+ 
      s  * Matrix([
           [0,-z,y],
           [z,0,-x],
           [-y,x,0]
           ])
       )

def vRot(V,theta,):
  return sp.diag(rot_vector(V,theta),1)  
  
Mx=xRot(phi)
Mz=zRot(theta)
Mt=tLat([dx,dy,dz])
#sp.pprint((Mt@Mz@Mx).subs(dict(dx=1,dy=2,dz=3,phi=pi/4,theta=pi/4)).evalf())
#sp.pprint(Mt.T)
#print([aij for aij in Mt.T])
column_major=False
def ij_to_flat(i,j):
  return i+j*4 if column_major else i*4+j
def flat_to_ij(l ):
  i,j=divmod(l,4)
  return  (j,i) if column_major else (i,j) 


def print_mMul(output=sys.stdout):
  print('export function mMul(')
  print('    ',
    ',\n     '.join(['['+', '.join([f'{c}{i}{j}' for l in range(16) for i,j in (flat_to_ij(l),)])+']' for c in 'ba']),
        ') {\n'
        '  const C = new Float32Array(16);',file=output,sep='')
  for l in range(16):
    i,j=flat_to_ij(l)
    print(f'  C[{l}] =',' + '.join([f'b{i}{k}*a{k}{j}' for k in range(4)]),';',file=output,sep='')
  print('  return C;\n'
        '}',file=output,sep='')
  
def write_ES6_module(output,column_major=False):
  def flatten(A):
    return [aij for aij in  (A.T if column_major else A)]
  def ij_to_l(i,j):#l is the index in the flattened matrix
    l=i+j*4 if column_major else i*4+j
    return l
  #comments & header
  if column_major:
    print( '// Column major matrix functions',file=output,sep='')
  else:  
    print('// Row major matrix functions', file=output,sep='')
  print(
    'const {sin,cos,sqrt,tan,PI}=Math,\n'
    '       pi=PI;\n',file=output,sep='')
  #x/y/z rotation matrix functions
  for fRot in (xRot,yRot,zRot):
    print(
      'export function '+fRot.__name__+'(a){\n'
      '  const s=sin(a);\n'
      '  const c=cos(a);\n'
      '  return ',flatten(fRot(phi).subs({sin(phi):"s",cos(phi):"c"})),';\n'#flatten sympy matrix
      '}\n',file=output,sep='')
  x,y,z,theta,s,c,c1=sp.symbols('x,y,z,theta,s,c,c1')
  print(
    'export function vRot([x,y,z],theta){\n'
    '  x??=0; y??=0; z??=0;\n'
    '  const length = sqrt(x*x + y*y + z*z);\n'
    '  if (length==0) {\n'
    '    if (theta===undefined){ \n'
    '       return [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];\n'
    '    }\n'
    '    else {\n'
    '       throw new Error("Rotation axis vector cannot be zero if a rotation angle is specified!");\n'
    '    }\n'
    '  }\n'
    '  if (theta===undefined) theta=length;\n'
    '  const c=cos(theta);\n'
    '  const c1=1-c;\n'
    '  const s=sin(theta);\n'
    '  x/=length;\n'
    '  y/=length;\n'
    '  z/=length;\n'
    '  return',flatten(vRot(Matrix([x,y,z]),theta).subs({sin(theta):s,(1-cos(theta)):c1,(-c1+1):c})),';\n'
    '}\n',file=output,sep='')
  #translation matrix functions
  print(
    'export function tLat([tx,ty,tz]){\n'
    '  tx??=0; ty??=0; tz??=0;\n'
    '  return ',flatten(tLat(['tx','ty','tz'])),';\n'#flatten sympy matrix
    '}\n',file=output,sep='')
  print(
    'export function scal([sx, sy, sz]) {\n'
    '  sx??=1; sy??=1; sz??=1;\n'
    '  return ',flatten(scal(['sx','sy','sz'])),';\n'
    '}\n',file=output,sep='')
  #transpose function
  T=['']*16
  for i in range(4):
    for j in range(4):
      T[ij_to_l(i,j)]=f'A[{ij_to_l(j,i)}]'
  print(
    'export function T(A){\n'
    '  return new Float32Array([',', '.join(T),']);\n'
    '}\n',file=output,sep='')
  #mMul function 
  print(
    'export function mMul(B,A){\n'
    '  const C=new Array(16);\n'
    '  let sum;\n'
    '  for (let i=0;i<4;++i)\n'
    '    for (let j=0;j<4;++j){\n'
    '      sum=0;\n'
    '      for (let k=0;k<4;++k)\n'
    '        sum+= B[',ij_to_l(*sp.symbols('i,k')),'] * A[',ij_to_l(*sp.symbols('k,j')),'];\n'
    '      C[',ij_to_l(*sp.symbols('i,j')),'] = sum;\n'
    '    }\n'
    '  return C;\n'
    '}\n',file=output,sep='')
    
  #vMul function
  print(
    'export function vMul(A,[x0,x1,x2,x3]){\n'
    '  x0??=0; x1??=0; x2??=0; x3??=0;\n'
    '  return new Float32Array([',', '.join(['+'.join([f'A[{ij_to_l(i,k)}]*x{k}' for k in range(4)])for i in range(4)]),']);\n'
    '}\n',file=output,sep='')
  
  fov,aspR,near,far,f,nfInv=sp.symbols('fov,aspR,near,far,f,nfInv')
  cot=sp.cot
  print(  
    'export function  persp(fov, aspR, near, far) {\n'
    '  const f = tan(pi * 0.5 - 0.5 * fov);\n'
    '  const nfInv = 1.0 / (near - far);\n'
    '  return ',flatten(persp(fov,aspR,near,far).subs({cot(fov/2):f,(near-far):1/nfInv})),';\n'
    '}\n',file=output,sep='')
  #functions to convert to row_major or column_major if not already in the correct order
  if column_major:
    print(
      'export function cMaj(A){return A;}\n'
      'export function rMaj(A){return T(A);}\n',file=output,sep='')
  else:
    print(
      'export function cMaj(A){return T(A);}\n'
      'export function rMaj(A){return A;}\n',file=output,sep='')
  
def camMat(targ,azim,elev,d):
  return tLat([0,0,-d])@xRot(elev-pi/2)@zRot(-azim-pi/2)@tLat(-sp.Matrix(targ))
def icamMat(targ,azim,elev,d):
  return tLat(sp.Matrix(targ))@zRot(-azim-pi/2).T@xRot(elev-pi/2).T@tLat([0,0,d])
a00,a01,a02,a10,a11,a12,a20,a21,a22=sp.symbols('a00,a01,a02,a10,a11,a12,a20,a21,a22')
def icamMat1(targ,camMat,d):
  return sp.Matrix(sp.BlockMatrix([[camMat[:3,:3].T,d*camMat[2,:3].T+sp.Matrix(targ)],[sp.zeros(1,3),sp.ones(1,1)]]))
  


def print_camMat(output=sys.stdout,column_major=True):
  tx,ty,tz,azim,elev,d=sp.symbols('tx,ty,tz,azim,elev,d')
  def flatten(A): return [aij for aij in  (A.T if column_major else A)]
  print(
    'export function camMat([tx,ty,tz],azim,elev,d){\n'
    '  // The function camMat calculates the camera matrix (similar to lookAt, but with different input parameters)\n'
    '  // tx,ty,tz: target coordinates\n'
    '  // azim: azimuth angle in radians\n'
    '  // elev: elevation angle in radians\n'
    '  // d: distance of camera from target. \n' 
    '  tx??=0; ty??=0; tz??=0; d??=0;\n'
    '  const s=sin(azim),\n'
    '        c=cos(azim),\n'
    '        se=sin(elev),\n'
    '        ce=cos(elev);\n'
    '  return new Float32Array(', 
             (flatten(camMat([tx,ty,tz],azim,elev,d)
             .subs({sin(elev):'se',cos(elev):'ce',sin(azim):'s',cos(azim):'c'}))),
             ')\n'
    '};\n',file=output,sep='')
    
def camPos(targ,camMat,dist):
  ex=camMat[2,0]
  ey=camMat[2,1]
  ez=camMat[2,2]
  tx,ty,tz,*_=targ
  return [tx+ex*dist,ty+ey*dist,tz+ez*dist,1]
     
def print_camPos(output=sys.stdout,column_major=True):
  def ij_to_l(i,j):#l is the index in the flattened matrix
    l=i+j*4 if column_major else i*4+j
    return l
  print(
    'export function camPos(targ,camMat,d){\n'
    '  //camera position in world coordinates'
    '  // tx,ty,tz: target coordinates\n'
    '  // camMat: camera matrix\n'
    '  // d: distance of camera from target. \n' 
    '  const [tx,ty,tz]=targ;\n'
    '  const ex=camMat[',ij_to_l(2,0),'], ey=camMat[',ij_to_l(2,1),'], ez=camMat[',ij_to_l(2,2),'];\n'
    '  return [tx+ex*d,ty+ey*d,tz+ez*d,1];\n'
    '};\n'
    ,file=output,sep='')
def print_icamMat(output=sys.stdout,column_major=True):
  def ij_to_l(i,j):#l is the index in the flattened matrix
    l=i+j*4 if column_major else i*4+j
    return l
  iC=['0']*16
  iC[15]='1'
  for i in range(3):
    for j in range(3):
      iC[ij_to_l(i,j)]=f'C[{ij_to_l(j,i)}]'#transposed 3x3 matrix is the inverse, because orthogonal and det=1
    iC[ij_to_l(i,3)]=f'C[{ij_to_l(2,i)}]*d+t[{i}]??0'
      
  print(
    'export function icamMat(t,C,d){\n'
    '  // The function icamMat calculates the inverse of the camera matrix for a given camera matrix\n'
    '  // t: target coordinates\n'
    '  // C: camera matrix\n'
    '  // d: distance of camera from target. \n' 
    '  d??=0;\n'
    '  return new Float32Array([', 
             ', '.join(iC),
             '])\n'
    '};\n',file=output,sep='')

#create_vfs_open creates a file in the virtual file system
cwd='/vfs/'
with create_vfs_open(vfs)(cwd+'m4_cMaj.js','w') as f:
  write_ES6_module(f,column_major=True)
  print_camMat(f,column_major=True)
  print_icamMat(f,column_major=True)
  print_camPos(f,column_major=True)
with create_vfs_open(vfs)(cwd+'m4_rMaj.js','w') as f:
  write_ES6_module(f,column_major=False)
  print_camMat(f,column_major=False)
  print_icamMat(f,column_major=False)
  print_camPos(f,column_major=False)
  
#s=f.getvalue()
#f.close()
#print(s)
#sp.pprint(mxzt.doit())
#sp.pprint(mxzt.doit().inv().simplify())
camera=dict(
	fov=30*deg, 
	target=[0,0,0], 
	azim=30*deg, 
	elev=40*deg,
	dist=1000
)
a={
    0:0.8660253882408142,
    1:-0.3830222189426422,
    10:-0.7660444378852844,
    11:0,
    12:0,
    13:0,
    14:-1000,
    15:1,
    2:-0.3213938176631927,
    3:0,
    4:-0.5,
    5:-0.663413941860199,
    6:-0.5566704273223877,
    7:0,
    8:0,
    9:0.6427876353263855,
}
b=[
    -0.4999999999999998,
    -0.5566703992264195,
    0.6634139481689384,
    0,
    0.8660254037844387,
    -0.3213938048432696,
    0.3830222215594889,
    0,
    0,
   0.766044443118978,
    0.6427876096865394,
    0,
    0,
    0,
    -1000,
    1
]
#sp.pprint(sp.Matrix([[a[i+j*4] for j in range(4)]for i in range(4)]))
#sp.pprint(sp.Matrix([[b[i+j*4] for j in range(4)]for i in range(4)]))
#sp.pprint(camMat(camera['target'],camera['azim'],camera['elev'],[0,0,-camera['dist']]).evalf())
tx,ty,tz,azim,elev,d=sp.symbols('tx,ty,tz,azim,elev,d')
cm=camMat([tx,ty,tz],azim,elev,d).subs({sin(elev):'se',cos(elev):'ce',sin(azim):'s',cos(azim):'c'})
icm=icamMat([tx,ty,tz],azim,elev,d).subs({sin(elev):'se',cos(elev):'ce',sin(azim):'s',cos(azim):'c'})
sp.pprint(cm)
sp.pprint(icm)
sp.pprint(icamMat1([tx,ty,tz],cm,d))
cm=sp.Matrix([[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]])
sp.pprint(icamMat1([tx,ty,tz],cm,d))

tx,ty,tz,azim,elev,d=0,0,0,camera['azim'],camera['elev'],camera['dist']
cm=camMat([tx,ty,tz],azim,elev,d)
icm=icamMat([tx,ty,tz],azim,elev,d)
sp.pprint((cm.inv()@sp.Matrix([0,0,0,1])).evalf())
print(camPos([tx,ty,tz],cm.evalf(),d))
print((cm@icm).evalf())

```

```python
with create_vfs_open(vfs)(cwd+'m4_cMaj.js','r') as f:
    print(f.read())
```

```python
mm=load_module('es6_html_to_iife_html','es6_html_to_iife_html.py',vfs)
```

```python
import es6_html_to_iife_html as es6iife
```

```python
es6iife.convert_es6_to_iife
```

```python
dir(es6iife)
```

```python
dir(mm)
```

```python
import BeautifulSoup

```

```python
!pip install BeautifulSoup

```

```python
!pip install BeautifulSoup4
```

```python

```

```python
import sys
```

```python
m=list(sys.modules.keys())
m.sort()
print(m)
```

```python

```
