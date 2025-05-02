---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

[![Run Jupyter Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RichardPotthoff/Notebooks/main?filepath=3Dpreview/3Dpreview.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RichardPotthoff/Notebooks/blob/main/3Dpreview/3Dpreview.ipynb)

+++

# 3D webgl JavaScript test

+++

## Virtual File System (VFS)
In order to keep all the files in one Jupyter notebook, a simple dict-based file system is used.
The SimpleVFS object provides an SimpleVFS.open() method that can be used to open files in this VFS.
Another method, "SimpleVFS.create_open()" returns an "open" function, that is compatible to "builtins.open" (it converts the "SimpleVFS.open" method into a proper function).

```{code-cell} ipython3
import sys
import io
import zipfile
import base64
from pathlib import PurePath

# Custom StringIO class to handle VFS updates
class VFSStringIO(io.StringIO):
    def __init__(self, vfs, path, initial_value='', mode='r'):
        super().__init__(initial_value)
        self.vfs = vfs
        self.path = path
        self.mode = mode
        if mode == 'a':
            self.seek(0, io.SEEK_END)

    def close(self):
        if self.mode in ('w', 'a'):
            self.vfs[self.path] = self.getvalue()
        super().close()

# SimpleVFS class with support for 'r', 'w', and 'a' modes
class SimpleVFS(dict):
    def __init__(self, mount_point='/', default_cwd='/', home_dir='/'):
        """
        Initialize the SimpleVFS with a mount point, default working directory, and home directory.

        Args:
            mount_point (str): The root of the virtual filesystem (default: '/').
            default_cwd (str, optional): The default current working directory (default: mount_point + '/').
            home_dir (str, optional): The virtual home directory for '~/...' paths (default: mount_point + '/home').
        """
        super().__init__()
        self.mount_point = PurePath(mount_point).as_posix()
        # Default default_cwd to mount_poin if not provided
        self.default_cwd = PurePath(default_cwd if default_cwd is not None else PurePath(self.mount_point)).as_posix()
        # Default home_dir to mount_point if not provided
        self.home_dir = PurePath(home_dir if home_dir is not None else PurePath(self.mount_point)).as_posix()
        # Validate that default_cwd and home_dir are under the mount point
        for path, name in [(self.default_cwd, 'default_cwd'), (self.home_dir, 'home_dir')]:
            if not path.startswith(self.mount_point):
                raise ValueError(f"{name.capitalize()} {path} must start with mount point {self.mount_point}")

    def _normalize_path(self, path):
        if not isinstance(path, str):
            raise TypeError(f"Path must be a string, got {type(path)}")
        normalized = PurePath(path).as_posix().lstrip('/')
        if not normalized:
            raise ValueError("Empty path is not allowed")
        return normalized

    def __setitem__(self, key, value):
        normalized_key = self._normalize_path(key)
        super().__setitem__(normalized_key, value)

    def __getitem__(self, key):
        normalized_key = self._normalize_path(key)
        return super().__getitem__(normalized_key)

    def open(self, path, mode='r', cwd=None):
        """
        Open a file in the virtual filesystem.

        Args:
            path (str): The path to the file. Supports './' (relative to cwd), '~/' (relative to home_dir),
                        absolute paths (starting with '/'), and relative paths.
            mode (str): The mode ('r', 'w', 'a').
            cwd (str, optional): The current working directory. Defaults to self.default_cwd.
        """
        if cwd is not None and not isinstance(cwd, str):
            raise TypeError(f"cwd must be a string or None, got {type(cwd)}")
        if cwd == '':
            raise ValueError("cwd cannot be an empty string")
        cwd = PurePath(cwd if cwd is not None else self.default_cwd)
        path = PurePath(path)

        # Handle '~/...' paths by replacing '~' with home_dir
        if path.as_posix().startswith('~/'):
            resolved_path = PurePath(self.home_dir) / path.as_posix()[2:]  # Skip '~/'
        else:
            # Check if the path is absolute or relative
            if path.is_absolute():
                resolved_path = path
            else:
                # Relative path: resolve against cwd
                resolved_path = cwd / path

        # Resolve the full path
        full_path = resolved_path
        full_path_str = full_path.as_posix()

        # Ensure the path starts with the mount point
        try:
            # This will raise a ValueError if full_path is not under mount_point
            relative_path = PurePath(full_path_str).relative_to(self.mount_point).as_posix()
        except ValueError:
            raise ValueError(f"Path {full_path_str} is outside the mount point {self.mount_point}")

        # The relative_path is the key for storage (already has leading '/' stripped by relative_to)
        normalized_path = relative_path
        if not normalized_path and mode != 'w':
            raise ValueError("Cannot open the root mount point in read or append mode")

        if mode == 'r':
            content = self.get(normalized_path)
            if content is None:
                raise FileNotFoundError(f"No such file: {normalized_path}")
            return VFSStringIO(self, normalized_path, initial_value=content, mode=mode)
        elif mode == 'w':
            return VFSStringIO(self, normalized_path, initial_value='', mode=mode)
        elif mode == 'a':
            content = self.get(normalized_path, '')
            return VFSStringIO(self, normalized_path, initial_value=content, mode=mode)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def create_open(self, cwd=None):
        effective_cwd = cwd if cwd is not None else self.default_cwd
        return lambda filename, mode='r': self.open(filename, mode, effective_cwd)
        
    def archive(self, filename_prefix="vfs_archive"):
        """
        Create a zip archive of all files in the VFS and return the raw bytes.
        
        Args:
            filename_prefix (str): Prefix for the archive filename (default: "vfs_archive").
        
        Returns:
            bytes: Raw bytes of the zip archive.
        """
        # Create a bytes buffer for the zip file
        zip_buffer = io.BytesIO()
        
        # Create a zip file in memory
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path, content in self.items():
                # Ensure file_path is a string and normalize it
                normalized_path = self._normalize_path(file_path)
                # Write each file to the zip with its original name
                zip_file.writestr(normalized_path, content)
        
        # Get the zip data as bytes and return
        return zip_buffer.getvalue()
```

## Module Loader
The function "load_module" makes it possible to load Python modules from the VFS without tampering with Python's built-in module loader.

```{code-cell} ipython3
def load_module(module_name, module_path=None, open=open):
    import importlib
    import sys
    if module_path is None:
        module_path = module_name + '.py'
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_loader(module_name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with open(module_path) as f:
        code = f.read()
        exec(code, module.__dict__)
    return module
```

```{code-cell} ipython3
vfs=SimpleVFS()
```

## ES6 JavaScript to IIFE converter and minifyer (Python module)

```{code-cell} ipython3
vfs['es6_html_to_iife_html.py']=r"""
import re
import os
from bs4 import BeautifulSoup
from collections import defaultdict
import re
open=open # used if embedded in Jupyter notebooks: gets replaced to use a virtual file system

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
delimiters=r'\[=({:<>;,?%&|*+-/' #removing ]}) from delimiters because of problems with asi not inserting semicolons if there is a \n behind the delimiter
whitespaces_to_right_of_delimiter =r'(?<=['+delimiters+r'])\s+'
whitespaces_to_left_of_delimiter =r'\s+(?=[]'+delimiters+'})'+r'])'
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
#        print(f'{full_path = }')
        imodule_dir=os.path.dirname(full_path)
        with open(full_path, 'r') as f:
           content = f.read()
        dependency_content += gather_dependencies(content, processed_modules, dependencies,in_process,module_dir=imodule_dir,module_filename=ifile_name, minify=minify)
    if module_filename:
      in_process.remove(module_filename)
    return dependency_content + converted
    
def convertES6toIIFE(content="import from './main.js';",module_dir='',module_filename='',minify=True):
  processed_modules = set()
  dependencies = defaultdict(set)
  iife_content = gather_dependencies(content, processed_modules, dependencies,  
                module_dir=module_dir, module_filename=module_filename,  minify=minify)
  return iife_content
                
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
    print("HTML processing completed with embedded ES6 modules converted to IIFE.")
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

## Load a Python module from the VFS

```{code-cell} ipython3
ES6converter=load_module('es6_html_to_iife_html','/es6_html_to_iife_html.py', vfs.create_open())
#Set the modules "open" function to out vfs.create_open():
ES6converter.open=vfs.create_open()
#(This 'tricks' the module into using our vfs for reading and writing files.)
```

```{code-cell} ipython3
dir(ES6converter)
```

## Code generator for a Matrix ES6 JavaScript module
A Python script that generates JavaScript code "m4_cMaj.js" + "m4_rMaj.js" for matrix operations. (using sympy)

```{code-cell} ipython3
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
#sp.pprint ((Mt@Mz@Mx).subs(dict(dx=1,dy=2,dz=3,phi=pi/4,theta=pi/4)).evalf())
#sp.pprint (Mt.T)
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
    '  //Camera position in world coordinates'
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
  
with vfs.open('m4_cMaj.js','w') as f:
  write_ES6_module(f,column_major=True)
  print_camMat(f,column_major=True)
  print_icamMat(f,column_major=True)
  print_camPos(f,column_major=True)
with vfs.open('m4_rMaj.js','w') as f:
  write_ES6_module(f,column_major=False)
  print_camMat(f,column_major=False)
  print_icamMat(f,column_major=False)
  print_camPos(f,column_major=False)
  
#s=f.getvalue()
#f.close()
#print(s)
#sp.pprint (mxzt.doit())
#sp.pprint (mxzt.doit().inv().simplify())

# some example calculations for testing javascript calculations:
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
#sp.pprint (sp.Matrix([[a[i+j*4] for j in range(4)]for i in range(4)]))
#sp.pprint (sp.Matrix([[b[i+j*4] for j in range(4)]for i in range(4)]))
#sp.pprint (camMat(camera['target'],camera['azim'],camera['elev'],[0,0,-camera['dist']]).evalf())
tx,ty,tz,azim,elev,d=sp.symbols('tx,ty,tz,azim,elev,d')
cm=camMat([tx,ty,tz],azim,elev,d).subs({sin(elev):'se',cos(elev):'ce',sin(azim):'s',cos(azim):'c'})
icm=icamMat([tx,ty,tz],azim,elev,d).subs({sin(elev):'se',cos(elev):'ce',sin(azim):'s',cos(azim):'c'})
print("camera matrix:")
sp.pprint (cm)
print()
print("inverse camera matrix:")
sp.pprint (icm)
print()
print("icamMat1([tx,ty,tz],cm,d):") 
sp.pprint (icamMat1([tx,ty,tz],cm,d))

cm=sp.Matrix([[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]])
print()
print("icamMat1([tx,ty,tz],aij,d):")
sp.pprint (icamMat1([tx,ty,tz],cm,d))

tx,ty,tz,azim,elev,d=0,0,0,camera['azim'],camera['elev'],camera['dist']

cm=camMat([tx,ty,tz],azim,elev,d)

icm=icamMat([tx,ty,tz],azim,elev,d)
print()
print("(cm.inv() @s p.Matrix([0,0,0,1]) ).evalf():")
sp.pprint ((cm.inv()@sp.Matrix([0,0,0,1])).evalf())
print()
print("camPos([tx,ty,tz],cm.evalf(),d):")
print(camPos([tx,ty,tz],cm.evalf(),d))
print()
print("check that (cm @ icm).evalf()  = I :")
sp.pprint ( (cm @ icm).evalf())
```

```{code-cell} ipython3
print(vfs["m4_cMaj.js"])
```

```{code-cell} ipython3
list(vfs.keys())
```

## HTML & JavaScript files

+++

### webgl-torus.html

```{code-cell} ipython3
vfs["webgl-torus.html"]=r"""
<!DOCTYPE html>
<html>
<head> 
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
<title>WebGL - 3D Torus</title>
</head>
<body></body>
<script src="./webgl-torus.js" type="module"></script>
</html>
"""
```

### webgl-torus.js

```{code-cell} ipython3
vfs["webgl-torus.js"]=r"""
if (typeof element === 'undefined') 
   window.element=document.body;
const deg=Math.PI / 180;
const iDeg=1/deg;
export function radToDeg(r) {
    return r * iDeg;
  }

export function degToRad(d) {
    return d * deg;
  }
 
export let vertexShaderSource=`
attribute vec4 a_position;
attribute vec4 a_normal;

uniform mat4 u_matrix;

varying vec4 v_color;

void main() {
  // Multiply the position by the matrix.
  gl_Position = u_matrix * a_position;

  // Pass the color to the fragment shader.
  v_color = vec4(0.0,0.0,0.0,1.0);  

  float wxp =max(a_normal.x,0.0);
  float wxn =max(-a_normal.x,0.0);
  float wyp= max(a_normal.y,0.0);
  float wyn= max(-a_normal.y,0.0);
  float wzp= max(a_normal.z,0.0);
  float wzn= max(-a_normal.z,0.0);
  
  v_color.xyz += wxp*wxp*wxp *vec3(1.0,0.0,0.0);
  v_color.xyz += wxn*wxn*wxn *vec3(0.1725,0.8157,0.7843);
  v_color.xyz += wyp*wyp*wyp *vec3(0.0,0.8706,0.0);
  v_color.xyz += wyn*wyn*wyn *vec3(0.9412,0.06275,1.0);
  v_color.xyz += wzp*wzp*wzp *vec3(0.102,0.3451,1.0);
  v_color.xyz += wzn*wzn*wzn *vec3(0.9137,0.9098,0.07451);

}
`;

export let fragmentShaderSource=`
precision mediump float;

// Passed in from the vertex shader.
varying vec4 v_color;

void main() {
   gl_FragColor = v_color;
}
`;
export const state={animate:true};
const hint=document.createElement("div");
hint.innerText="Colors indicate surface normal directions: +xyz=rgb, -xyz=cmy \nDrag to rotate the view.";
element.appendChild(hint);
const canvas = document.createElement("canvas");
canvas.style.display="inline-block";
canvas.style.width="400px";
canvas.style.height="400px";
element.appendChild(canvas);
const  gl = canvas.getContext("webgl");

let isDragging = false;
let previousTouchX = 0;
let previousTouchY = 0;
//let rotationX = 0;
//let rotationY = 0;
let scale = 1;
let translateX = 0;
let translateY = 0;

// Touch Start
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault(); // Prevent scrolling
    isDragging = true;
    
    // Store initial touch position
    if (e.touches.length === 1) {
        previousTouchX = e.touches[0].clientX;
        previousTouchY = e.touches[0].clientY;
    }
});

// Touch Move
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    
    if (!isDragging) return;

    if (e.touches.length === 1) {
        // Single finger - Rotation
        const currentX = e.touches[0].clientX;
        const currentY = e.touches[0].clientY;
        
        // Calculate movement delta
        const deltaX = currentX - previousTouchX;
        const deltaY = currentY - previousTouchY;
        
        // Update rotation (adjust sensitivity as needed)
        camera.azim = (camera.azim-deltaX * 0.01) % (Math.PI*2);
        camera.elev = Math.max(Math.min(camera.elev+deltaY * 0.01,Math.PI/2),-Math.PI/2);
        
        // Update previous position
        previousTouchX = currentX;
        previousTouchY = currentY;
        
    } else if (e.touches.length === 2) {
        // Two fingers - Zoom and Pan
        
        // Calculate distance between touches
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        const currentDistance = Math.hypot(
            touch1.clientX - touch2.clientX,
            touch1.clientY - touch2.clientY
        );
        
        // Store initial distance on first two-finger touch
        if (!canvas.dataset.previousDistance) {
            canvas.dataset.previousDistance = currentDistance;
        }
        
        // Zoom calculation
        const previousDistance = parseFloat(canvas.dataset.previousDistance);
        const distanceDelta = currentDistance - previousDistance;
        scale += distanceDelta * 0.005;
        scale = Math.max(0.1, Math.min(scale, 10)); // Limit zoom range
        
        // Pan calculation (using midpoint movement)
        const midX = (touch1.clientX + touch2.clientX) / 2;
        const midY = (touch1.clientY + touch2.clientY) / 2;
        
        if (canvas.dataset.previousMidX) {
            translateX += midX - parseFloat(canvas.dataset.previousMidX);
            translateY += midY - parseFloat(canvas.dataset.previousMidY);
        }
        
        // Store current values for next frame
        canvas.dataset.previousDistance = currentDistance;
        canvas.dataset.previousMidX = midX;
        canvas.dataset.previousMidY = midY;
    }
    
    // Apply transformations to your 3D object here
    if (!state.animate)
	  drawScene();
});

// Touch End
canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    isDragging = false;
    // Clean up stored values
    delete canvas.dataset.previousDistance;
    delete canvas.dataset.previousMidX;
    delete canvas.dataset.previousMidY;
});

// Touch Cancel (in case touch is interrupted)
canvas.addEventListener('touchcancel', () => {
    isDragging = false;
    delete canvas.dataset.previousDistance;
    delete canvas.dataset.previousMidX;
    delete canvas.dataset.previousMidY;
});

// Optional: Prevent default touch behavior on the document
document.addEventListener('touchmove', (e) => {
    if (e.target === canvas) {
        e.preventDefault();
    }
}, { passive: false });

export let program;
export let normalLocation;
export let positionLocation;
export let matrixLocation;
//export let cameraDistance;
export const camera={
	fov:30*deg, 
	target:[0,0,0], 
	azim:30*deg, 
	elev:40*deg,
	dist:1000
};

  
function cube(dx=1,dy,dz){
  const numVertices=24;
  const stride=6*4;
  const X=0.5*dx;
  const Y=dy|X;
  const Z=dz|X;
  const vertices= new Float32Array([
    X,-Y,-Z, 1, 0, 0, X, Y,-Z, 1, 0, 0, X, Y, Z, 1, 0, 0, X,-Y, Z, 1, 0, 0,//+X
   -X, Y, Z,-1, 0, 0,-X, Y,-Z,-1, 0, 0,-X,-Y,-Z,-1, 0, 0,-X,-Y, Z,-1, 0, 0,//-X
   -X, Y,-Z, 0, 1, 0,-X, Y, Z, 0, 1, 0, X, Y, Z, 0, 1, 0, X, Y,-Z, 0, 1, 0,//+Y
    X,-Y, Z, 0,-1, 0,-X,-Y, Z, 0,-1, 0,-X,-Y,-Z, 0,-1, 0, X,-Y,-Z, 0,-1, 0,//-Y
   -X,-Y, Z, 0, 0, 1, X,-Y, Z, 0, 0, 1, X, Y, Z, 0, 0, 1,-X, Y, Z, 0, 0, 1,//+Z
    X, Y,-Z, 0, 0,-1, X,-Y,-Z, 0, 0,-1,-X,-Y,-Z, 0, 0,-1,-X, Y,-Z, 0, 0,-1,//-Z
  ]);
  const indices=new Int16Array([
   0, 1, 2, 2, 3, 0,//+X
   4, 5, 6, 6, 7, 4,//-X
   8, 9,10,10,11, 8,//+Y
  12,13,14,14,15,12,//-Y
  16,17,18,18,19,16,//+Z
  20,21,22,22,23,20 //-Z
  ]);
  return {indices,vertices,stride,numVertices};
}
  
function circle(r,n){
	  const epath=[]
	  for (let i=0;i<n;i++){
	    const theta=i*2*Math.PI/n;
		const s=Math.sin(theta);
		const c=Math.cos(theta);
		epath[i] = [[r*c,r*s],[-s,c]];//x
	}
	return epath;
}
function extrude(epath,shape){
	const m=epath.length;
	const n=shape.length;
	const numVertices=m*n;
	const vertices=new Float32Array(numVertices*6);//xyz dxdydz (coordinate and direction vector)
	const stride=6*4;//xyz dxdydz float32
	const indices=new Int16Array(numVertices*6);//2 triangles per vertex
//	console.log(numVertices)
	for (let j=0;j<m;j++){
		let [[x_p,y_p],[ms_p,c_p]]=epath[j];
		for (let i=0;i<n;i++){
		    const [[x_s,y_s],[ms_s,c_s]]=shape[i];
			const k=j*n+i
			vertices[k*6 + 0]= x_p+x_s*c_p;
			vertices[k*6 + 1]= y_p-x_s*ms_p;
            vertices[k*6 + 2]= y_s; //z is the y direction of the shape
			vertices[k*6 + 3]= c_s*c_p;
			vertices[k*6 + 4]= -c_s*ms_p;
			vertices[k*6 + 5]= -ms_s;
			indices[k*6+0]=  j           * n +   i          ; //j  ,i  
			indices[k*6+1]=((j + 1) % m) * n + ((i + 1) % n); //j+1,i+1
			indices[k*6+2]=  j           * n + ((i + 1) % n); //j  ,i+1
			indices[k*6+3]=  j           * n +   i          ; //j  ,i  
			indices[k*6+4]=((j + 1) % m) * n +   i          ; //j+1,i
			indices[k*6+5]=((j + 1) % m) * n + ((i + 1) % n); //j+1,i+1
		}
	}
	return {indices,vertices,stride,numVertices}
}

export const ShapeData={};

function initialize() {
  Object.assign(ShapeData,(extrude(circle(150,50),circle(50.0,30))));
	//  ShapeData=cube(150);
//  console.log(ShapeData.vertices.slice(0, 24));
  if (!gl) {
    return;
  }
  // setup GLSL program
  const vertexShader=gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertexShader, vertexShaderSource);
  gl.compileShader(vertexShader);
  const fragmentShader=gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragmentShader,fragmentShaderSource);
  gl.compileShader(fragmentShader);
  program = gl.createProgram();
  [vertexShader,fragmentShader].forEach(function(shader) {
      gl.attachShader(program, shader);
    });
  gl.linkProgram(program);
  gl.useProgram(program);
  // look up where the vertex data needs to go.
  positionLocation = gl.getAttribLocation(program, "a_position");
  normalLocation = gl.getAttribLocation(program, "a_normal");
  // lookup uniforms
  matrixLocation = gl.getUniformLocation(program, "u_matrix");
  // Create a buffer to put positions in
  let vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER,ShapeData.vertices , gl.STATIC_DRAW);
  //set vertex position attributes
  gl.enableVertexAttribArray(positionLocation);   
  gl.vertexAttribPointer(positionLocation,3,gl.FLOAT,false,ShapeData.stride,0);
  // set the color attribute
  gl.enableVertexAttribArray(normalLocation);
  gl.vertexAttribPointer(normalLocation,3,gl.FLOAT,true,ShapeData.stride,12);
  let indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ShapeData.indices, gl.STATIC_DRAW);
  requestAnimationFrame(drawScene);
}

  
  // Draw the scene.
  export function drawScene() {
	let targ=[0,0,0];
	let cM=M4.camMat(targ,camera.azim,camera.elev,camera.dist);
//	console.log("camMat",cM);
	let icM=M4.icamMat(targ,cM,camera.dist);
//    console.log("icamMat",icM);
//	console.log("vMul(icM,[0,0,0,1])",M4.vMul(icM,[0,0,0,1]))
	let cP=M4.camPos(targ,cM,camera.dist);
//	console.log("camPos",cP);
//	console.log("vRot",M4.vRot([0,0,camera.azim]));
	

	
    gl.canvas.width=gl.canvas.clientWidth;
	gl.canvas.height=gl.canvas.clientHeight;
    // Tell WebGL how to convert from clip space to pixels
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    // Clear the canvas AND the depth buffer.
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Turn on culling. By default backfacing triangles
    // will be culled.
    gl.enable(gl.CULL_FACE);

    // Enable the depth buffer
    gl.enable(gl.DEPTH_TEST);

    // Tell it to use our program (pair of shaders)
    gl.useProgram(program);

    // Compute the projection matrix
    let aspR = gl.canvas.clientWidth / gl.canvas.clientHeight;
    let zNear = 1;
    let zFar = camera.dist+1000;
    let projectionMatrix = M4.persp(camera.fov, aspR, zNear, zFar);
	
    // create a viewProjection matrix. This will both apply perspective
    // AND move the world so that the camera is effectively the origin
    let viewProjectionMatrix = M4.mMul(projectionMatrix,cM);
	drawTorus([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], viewProjectionMatrix, matrixLocation);
	if (state.animate){
	  //console.log(`Animating, rot: ${parseFloat(slider.value).toFixed(0)} deg`);
      camera.azim = (camera.azim+0.02)%(Math.PI*2);
      //slider.value = (camera.azim/deg);
      requestAnimationFrame(drawScene);
    }
  }

  function drawTorus(matrix, viewProjectionMatrix, matrixLocation) {
    // multiply that with the viewProjecitonMatrix
    matrix = M4.mMul(viewProjectionMatrix, matrix);
    // Set the matrix.
    gl.uniformMatrix4fv(matrixLocation, false, M4.cMaj(matrix));

    // Draw the geometry.
    gl.drawElements(gl.TRIANGLES, ShapeData.indices.length, gl.UNSIGNED_SHORT, 0);

  }
export {drawScene as render};
//import * as m4 from './m4.js';
//import * as M4 from './M4.js';
import * as M4 from './m4_cMaj.js';
//let {addVectors,subtractVectors,normalize,cross}=m4;
initialize();
"""
```

### m4_cMaj.js
(code generated by the Python script above)

```{code-cell} ipython3
print(vfs['m4_cMaj.js'])
```

##

```{code-cell} ipython3
list(vfs.keys())
```

## Conversion ES6 -> IIFE

```{code-cell} ipython3
from time import perf_counter
print(f'{(len1:=len(vfs["webgl-torus.js"]))=}\n{(len2:=len(vfs["m4_cMaj.js"])) = }\n{len1+len2 = }')
print()
t1=perf_counter()
iife_script=ES6converter.convertES6toIIFE('import "./webgl-torus.js";',minify=True)
print("HTML processing completed with ES6 modules converted to IIFE functions.")
t2=perf_counter()
print()
print(f'{t2-t1=}s')
print()
print(f'{len(iife_script)=}')
print()
print('IIFE JavaScript:\n',iife_script[:500], ' ...') 
#converting the html file with ES6 module references into a single html file with minified iife functions.   
print()
print('"webgl-torus.html" with ES6 module references  -->  stand alone minified "index.html" ')
ES6converter.process_html('webgl-torus.html',minify=True,output_file='index.html')
```

```{code-cell} ipython3
#copy index.html from virtual file system to disk
#with open('index.html','w') as f: f.write(vfs['index.html'])
```

## Interactive 3D Viewer
An ipywidget.Output widget is used in combination with IPython.Display.JavaScript to show the interactive webgl viewer.

```{code-cell} ipython3
from IPython.display import display,Javascript
import ipywidgets as widgets
output=widgets.Output()
with output:
    display(Javascript(iife_script))
display(output)
```

```{code-cell} ipython3
#display the same output widget again: A second, independent view.
display(output)
display(Javascript("let {camera,render,state,degToRad}=modules['webgl-torus.js'];" 
                   "camera.elev=degToRad(-20);"
                   "camera.azim=degToRad(-45);"  
                   "state.animate=false;"
                   "requestAnimationFrame(render);"
                   ))
```

```{code-cell} ipython3
display(Javascript("let {camera,render,state,degToRad}=modules['webgl-torus.js'];" 
                   "requestAnimationFrame(render);"
                   ))
```

```{code-cell} ipython3
import ipywidgets as widgets
from IPython.display import display, HTML
import base64

class DownloadButton(widgets.Button):
    def __init__(self, data_callback, filename="download.zip", **kwargs):
        """
        Create a button to download a zip archive generated by a callback function.
        
        Args:
            data_callback: A function that returns the data (as bytes) to be downloaded.
            filename: The name of the file to be downloaded (default: "download.zip").
            **kwargs: Additional arguments for the widgets.Button constructor.
        """
        super().__init__(description="Download Archive", **kwargs)
        self.data_callback = data_callback
        self.filename = filename
        self.output = widgets.Output()
        self.on_click(self._on_click)

    def _on_click(self, b):
        # Get raw bytes from the callback
        data_bytes = self.data_callback()
        # Encode to base64
        b64_data = base64.b64encode(data_bytes).decode()
        # Generate HTML link for download
        download_link = f'<a download="{self.filename}" href="data:application/zip;base64,{b64_data}" style="display:block;">Click to download {self.filename}</a>'
        with self.output:
            self.output.clear_output()
            display(HTML(download_link))

    def display(self):
        """Display the button and its output widget."""
        display(self)
        display(self.output)


# Create and display the download button
download_btn = DownloadButton(data_callback=lambda: vfs.archive(filename_prefix="3Dpreview"), filename="3Dpreview.zip")
download_btn.display()
```

```{code-cell} ipython3
#save complete contents of vfs as zip file to disk:
#with open('3Dpreview.zip','wb') as f: f.write(vfs.archive(filename_prefix='3Dpreview'))  
```

# Trackpad example

+++

## Source files

+++

### trackpad.html

```{code-cell} ipython3
vfs["trackpad.html"]=r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
  <title>Trackpad Test</title>
</head>
<body>
  <script src="main.js" type="module"></script>
</body>
</html>
"""
```

### main.js

```{code-cell} ipython3
vfs["main.js"]=r"""
// main.js
import { StateManager } from './stateManager.js';
import { Trackpad } from './trackpad.js';
import { Sliders } from './sliders.js';

// Detect if we're in Jupyter
const isJupyter = typeof element !== 'undefined';

// Configuration
const config = { minX: -40, maxX: 40, minY: -30, maxY: 30, width: 400, height: 300 };

// Create or reuse StateManager
const stateManager = new StateManager();

// Create and append canvas
const canvas = document.createElement('canvas');
canvas.width = config.width;
canvas.height = config.height;
canvas.style.border = '1px solid black';
const container = isJupyter ? element : document.body;
container.appendChild(canvas);
const ctx = canvas.getContext('2d');

// Initialize state
const initialState = { x: 0, y: 0 };
stateManager.setState(initialState);

// Initialize Trackpad
const trackpad = new Trackpad({
  element: canvas,
  onPositionChange: (position) => {
    stateManager.setState({ x: position.x, y: position.y });
  },
  minX: config.minX,
  maxX: config.maxX,
  minY: config.minY,
  maxY: config.maxY
});

// Render function
const render = () => {
  const state = stateManager.getState();
  const [canvasX, canvasY] = trackpad.scaleToCanvas(state.x, state.y);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);
  ctx.fillStyle = 'blue';
  ctx.fill();
};

// UUIDs for Jupyter widgets
export const uuids = {
  x_slider: null,
  y_slider: null,
  state_text: null
};

// Initialize function for Jupyter
export function initialize({state_text_uuid}) {
  if (!isJupyter) return; // Only run in Jupyter

  // Access widgets by UUID
  const stateText = document.querySelector(`.${state_text_uuid} input`);

  if (!stateText) {
    console.error('Could not find Jupyter widgets');
    return;
  }

  // Sync StateManager with stateText
  function updateStateText() {
    const state = stateManager.getState();
    stateText.value = JSON.stringify(state);
  }

  stateManager.setState(JSON.parse(stateText.value));

  stateManager.subscribe((state) => {
    updateStateText();
  });

  const observer = new MutationObserver(() => {
    const newState = JSON.parse(stateText.value);
    stateManager.setState(newState);
  });
  observer.observe(stateText, { attributes: true, attributeFilter: ['value'] });
}

// Initialize UI based on environment
if (!isJupyter) {
  // Standalone HTML: Use HTML sliders
  new Sliders(stateManager, config);
  stateManager.subscribe(render);
  render();
} else {
  // Jupyter: Wait for initialize() to be called after UUIDs are set
  stateManager.subscribe(render);
  render();
}
"""
```

### trackpad.js

```{code-cell} ipython3
vfs["trackpad.js"]=r"""
// trackpad.js
export class Trackpad {
  constructor({
    element,
    onPositionChange,
    minX = -40,
    maxX = 40,
    minY = -30,
    maxY = 30
  }) {
    this.element = element; // External element (e.g., canvas) for input
    this.onPositionChange = onPositionChange; // Callback for position updates
    this.config = { minX, maxX, minY, maxY }; // Configurable ranges
    this.isDragging = false;
    this.currentCanvasX = 0;
    this.currentCanvasY = 0;
    this.bindEvents();
  }

  bindEvents() {
    const start = (e, touch = false) => {
      e.preventDefault();
      this.isDragging = true;
      this.updatePosition(e, touch);
    };

    const move = (e, touch = false) => {
      e.preventDefault();
      if (!this.isDragging) return;
      this.updatePosition(e, touch);
    };

    const end = () => {
      this.isDragging = false;
    };

    // Mouse events
    this.element.addEventListener('mousedown', (e) => start(e));
    this.element.addEventListener('mousemove', (e) => move(e));
    this.element.addEventListener('mouseup', end);
    this.element.addEventListener('mouseleave', end);

    // Touch events
    this.element.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) start(e, true);
    }, { passive: false });
    this.element.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1) move(e, true);
    }, { passive: false });
    this.element.addEventListener('touchend', end);
    this.element.addEventListener('touchcancel', end);
  }

  updatePosition(e, touch = false) {
    const rect = this.element.getBoundingClientRect();
    let clientX = touch ? e.touches[0].clientX : e.clientX;
    let clientY = touch ? e.touches[0].clientY : e.clientY;
    const canvasX = clientX - rect.left;
    const canvasY = clientY - rect.top;
    this.currentCanvasX = Math.max(0, Math.min(canvasX, this.element.clientWidth));
    this.currentCanvasY = Math.max(0, Math.min(canvasY, this.element.clientHeight));
    const [scaledX, scaledY] = this.scaleFromCanvas(this.currentCanvasX, this.currentCanvasY);
    this.onPositionChange({ x: scaledX, y: scaledY });
  }

  scaleFromCanvas(canvasX, canvasY) {
    const { minX, maxX, minY, maxY } = this.config;
    const width = this.element.clientWidth;
    const height = this.element.clientHeight;
    const x = minX + (canvasX / width) * (maxX - minX);
    const y = minY + ((height - canvasY) / height) * (maxY - minY); // Invert Y-axis
    return [x, y];
  }

  scaleToCanvas(x, y) {
    const { minX, maxX, minY, maxY } = this.config;
    const width = this.element.clientWidth;
    const height = this.element.clientHeight;
    const canvasX = ((x - minX) / (maxX - minX)) * width;
    const canvasY = height - ((y - minY) / (maxY - minY)) * height; // Invert Y-axis
    return [canvasX, canvasY];
  }
}
"""
```

### slider.js

```{code-cell} ipython3
vfs["sliders.js"]=r"""
// sliders.js
export class Sliders {
  constructor(stateManager, config) {
    this.stateManager = stateManager;
    this.config = config;
    this.initialize();
  }

  initialize() {
    this.container = document.createElement('div');
    document.body.appendChild(this.container);

    // X slider with label and value display (mimicking ipywidgets)
    const xWrapper = document.createElement('div');
    xWrapper.style.display = 'flex';
    xWrapper.style.alignItems = 'center';
    xWrapper.style.margin = '5px 0';

    const xLabel = document.createElement('span');
    xLabel.textContent = 'X: ';
    xLabel.style.width = '50px';
    xWrapper.appendChild(xLabel);

    this.xSlider = document.createElement('input');
    this.xSlider.type = 'range';
    this.xSlider.min = this.config.minX;
    this.xSlider.max = this.config.maxX;
    this.xSlider.step = '0.1';
    this.xSlider.style.width = '200px';
    xWrapper.appendChild(this.xSlider);

    this.xValue = document.createElement('span');
    this.xValue.style.marginLeft = '10px';
    this.xValue.style.width = '50px';
    xWrapper.appendChild(this.xValue);

    this.container.appendChild(xWrapper);

    // Y slider with label and value display
    const yWrapper = document.createElement('div');
    yWrapper.style.display = 'flex';
    yWrapper.style.alignItems = 'center';
    yWrapper.style.margin = '5px 0';

    const yLabel = document.createElement('span');
    yLabel.textContent = 'Y: ';
    yLabel.style.width = '50px';
    yWrapper.appendChild(yLabel);

    this.ySlider = document.createElement('input');
    this.ySlider.type = 'range';
    this.ySlider.min = this.config.minY;
    this.ySlider.max = this.config.maxY;
    this.ySlider.step = '0.1';
    this.ySlider.style.width = '200px';
    yWrapper.appendChild(this.ySlider);

    this.yValue = document.createElement('span');
    this.yValue.style.marginLeft = '10px';
    this.yValue.style.width = '50px';
    yWrapper.appendChild(this.yValue);

    this.container.appendChild(yWrapper);

    const initialState = this.stateManager.getState();
    this.xSlider.value = initialState.x;
    this.ySlider.value = initialState.y;
    this.xValue.textContent = initialState.x.toFixed(1);
    this.yValue.textContent = initialState.y.toFixed(1);

    this.bindEvents();

    this.stateManager.subscribe((state) => {
      this.xSlider.value = state.x;
      this.ySlider.value = state.y;
      this.xValue.textContent = state.x.toFixed(1);
      this.yValue.textContent = state.y.toFixed(1);
    });
  }

  bindEvents() {
    this.xSlider.addEventListener('input', () => {
      const state = this.stateManager.getState();
      this.stateManager.setState({ ...state, x: parseFloat(this.xSlider.value) });
    });

    this.ySlider.addEventListener('input', () => {
      const state = this.stateManager.getState();
      this.stateManager.setState({ ...state, y: parseFloat(this.ySlider.value) });
    });
  }
}
"""
```

### stateManager.js

```{code-cell} ipython3
vfs["stateManager.js"]=r"""
// stateManager.js
export class StateManager {
  constructor() {
    this.state = {};
    this.subscribers = new Set();
  }

  getState() {
    return { ...this.state };
  }

  setState(newState) {
    // Only update if the state has actually changed
    const hasChanged = Object.keys(newState).some(key => this.state[key] !== newState[key]);
    if (!hasChanged) return;

    this.state = { ...this.state, ...newState };
    this.notifySubscribers();
  }

  subscribe(callback) {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  notifySubscribers() {
    this.subscribers.forEach(callback => callback(this.state));
  }
}
"""
```

### Minification -> index2.html

```{code-cell} ipython3
ES6converter.process_html('trackpad.html',minify=True,output_file='index2.html')
#copy index2.html from virtual file system to disk
with open('index2.html','w') as f: f.write(vfs['index2.html'])
```

## Jupyter integration of Trackpad

```{code-cell} ipython3
import ipywidgets as widgets
from IPython.display import Javascript, display
import uuid
import json

state={"x":0,"y":0}
# Create widgets for the trackpad
x_slider = widgets.FloatSlider(value=0, min=-40, max=40, step=0.1, description='X:')
y_slider = widgets.FloatSlider(value=0, min=-30, max=30, step=0.1, description='Y:')
state_text = widgets.Text(value=json.dumps(state), disabled=True, layout={'display': 'none'})

# Generate UUIDs with prefix
state_text_id = "uuid" + str(uuid.uuid4()).replace('-', '')

# Assign UUIDs to widgets
state_text.add_class(state_text_id)

# Display widgets
output = widgets.Output()
display(output, x_slider, y_slider, state_text)

# Inject JavaScript to set UUIDs and initialize
print()

iife_script2=ES6converter.convertES6toIIFE('import "./main.js";',minify=True)
js_code = iife_script2 + "window.modules['main.js'].initialize({state_text_id:'"+state_text_id+"'});"

with output:
    display(Javascript(js_code))

# Python handler for state changes
def on_x_slider_change(change):
    state=json.loads(state_text.value)
    state['x'] = change['new']
    state_text.value = json.dumps(state)
    
def on_y_slider_change(change):
    state=json.loads(state_text.value)
    state['x'] = change['new']
    state_text.value = json.dumps(state)

def on_state_change(change):
    state = json.loads(change['new'])
    x_slider.value = state['x']
    y_slider.value = state['y']

x_slider.observe(on_x_slider_change, names='value')
y_slider.observe(on_y_slider_change, names='value')
state_text.observe(on_state_change, names='value')
```

```{code-cell} ipython3
def on_state_change(change):
    state = eval(change['new'])  # Parse JSON string to dict
    x_slider.value = state['x']
    y_slider.value = state['y']

state_text.observe(on_state_change, names='value')
```
