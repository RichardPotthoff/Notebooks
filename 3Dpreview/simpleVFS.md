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

```{code-cell} ipython3
import sys
import io
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

```{code-cell} ipython3
ES6converter=load_module('es6_html_to_iife_html','/es6_html_to_iife_html.py', vfs.create_open())
```

```{code-cell} ipython3
dir(ES6converter)
```

```{code-cell} ipython3

```
