import marimo

__generated_with = "0.13.10"
app = marimo.App()

with app.setup:
    # Initialization code that runs before all other cells
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    [![Run Jupyter Notebooks](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RichardPotthoff/Notebooks/main?filepath=3Dpreview/3Dpreview.ipynb)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RichardPotthoff/Notebooks/blob/main/3Dpreview/3Dpreview.ipynb)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 3D webgl JavaScript test""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Virtual File System (VFS)
    In order to keep all the files in one Jupyter notebook, a simple dict-based file system is used.
    The SimpleVFS object provides an SimpleVFS.open() method that can be used to open files in this VFS.
    Another method, "SimpleVFS.create_open()" returns an "open" function, that is compatible to "builtins.open" (it converts the "SimpleVFS.open" method into a proper function).
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from time import perf_counter
    import os
    import sys
    import io
    import zipfile
    import base64
    from pathlib import PurePath
    from IPython.display import Javascript, display
    import ipywidgets as widgets
    import anywidget
    import traitlets
    from cmath import inf,exp,pi
    import re

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
        @property
        def open(self):
            return self.open_with_cwd()

        def open_with_cwd(self, cwd=None):
            effective_cwd = cwd if cwd is not None else self.default_cwd
            return lambda *args,**kwargs: self._open(*args, **(dict(cwd=effective_cwd) | kwargs))

        def _open(self, path, mode='r', cwd=None):
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
    return (
        SimpleVFS,
        anywidget,
        base64,
        exp,
        inf,
        perf_counter,
        pi,
        sys,
        traitlets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Module Loader
    The function "load_module" makes it possible to load Python modules from the VFS without tampering with Python's built-in module loader.
    """
    )
    return


@app.cell
def _(sys):
    def load_module(module_name, code):
        import importlib
        #if module_name in sys.modules:
        #   return sys.modules[module_name]
        spec = importlib.util.spec_from_loader(module_name, None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        exec(code, module.__dict__)
        return module
    return (load_module,)


@app.cell
def _(SimpleVFS):
    vfs=SimpleVFS()
    return (vfs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## ES6 JavaScript to IIFE converter and minifyer (Python module)""")
    return


@app.cell
def _(vfs):
    es6_html_to_iife_html_code=r"""
    import re
    import os
    from collections import defaultdict

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
      export_pattern = r'(?=^|;)\s*(export\s+(?P<export_default>default\s+)?(?P<export_type>(?:async\s+)?(?:function|const|let|var|class)(?:\s+|\s*\*\s*))?(?P<export_name>\w+)\s*)'

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

    def gather_dependencies(content, processed_modules, dependencies, in_process=None, module_dir=None, module_filename=None, minify=False,open=open):
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
            dependency_content += gather_dependencies(content, processed_modules, dependencies,in_process,module_dir=imodule_dir,module_filename=ifile_name, minify=minify,open=open)
        if module_filename:
          in_process.remove(module_filename)
        return dependency_content + converted

    def convertES6toIIFE(content="import from './main.js';",module_dir='',module_filename='',minify=True,open=open):
      processed_modules = set()
      dependencies = defaultdict(set)
      iife_content = gather_dependencies(content, processed_modules, dependencies,  
                    module_dir=module_dir, module_filename=module_filename,  minify=minify,open=open)
      return iife_content

    def process_html(html_path,minify=False,output_file='output.html',open=open):
        from bs4 import BeautifulSoup
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
                    module_dir=module_dir, module_filename=module_filename,  minify=minify,open=open)
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

    vfs['es6_html_to_iife_html.py']=es6_html_to_iife_html_code
    return (es6_html_to_iife_html_code,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load a Python module from the VFS""")
    return


@app.cell
def _(es6_html_to_iife_html_code, load_module):
    ES6converter=load_module('es6_html_to_iife_html', es6_html_to_iife_html_code)
    #Set the modules "open" function to out vfs.create_open():
    #ES6converter.open=vfs.create_open()
    #(This 'tricks' the module into using our vfs for reading and writing files.)
    return (ES6converter,)


@app.cell
def _(ES6converter):
    dir(ES6converter)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Code generator for a Matrix ES6 JavaScript module
    A Python script that generates JavaScript code "m4_cMaj.js" + "m4_rMaj.js" for matrix operations. (using sympy)
    """
    )
    return


@app.cell
def _(mo):
    generate_m4_js_btn=mo.ui.run_button(label="generate m4_cMaj.js")
    return (generate_m4_js_btn,)


@app.cell
def _(generate_m4_js_btn, mo, sys, vfs):
    mo.stop(not generate_m4_js_btn.value,generate_m4_js_btn)

    def _():
        import sympy as sp

        (theta, phi, d, dx, dy, dz, x_t, y_t, z_t) = sp.symbols(
            "theta,phi,d,dx,dy,dz,x_t,y_t,z_t"
        )
        Matrix = sp.Matrix
        sin = sp.sin
        cos = sp.cos
        tan = sp.tan
        pi = sp.pi
        pi_2 = sp.pi / 2
        deg = sp.pi / 180

        def tLat(v):
            (dx, dy, dz, *_) = v
            return sp.Matrix(
                sp.BlockMatrix(
                    [
                        [sp.eye(3), sp.Matrix([dx, dy, dz])],
                        [sp.zeros(1, 3), sp.ones(1, 1)],
                    ]
                )
            )

        def xRot(alpha):
            return sp.diag(sp.rot_axis1(-alpha), 1)

        def yRot(alpha):
            return sp.diag(sp.rot_axis2(-alpha), 1)

        def zRot(alpha):
            return sp.diag(sp.rot_axis3(-alpha), 1)

        def scal(v):
            (sx, sy, sz, *_) = v
            return sp.diag(sx, sy, sz, 1)

        def persp(fov, aspR, near, far):
            _f = tan(pi / 2 - fov / 2)
            nfInv = 1 / (near - far)
            return Matrix(
                [
                    [_f / aspR, 0, 0, 0],
                    [0, _f, 0, 0],
                    [0, 0, (near + far) * nfInv, near * far * nfInv * 2],
                    [0, 0, -1, 0],
                ]
            )

        def rot_vector(V, theta=None):
            absV = abs(V)
            if theta == None:
                theta = absV
            (x, y, z, *_) = V
            c = cos(theta)
            s = sin(theta)
            v = Matrix([x, y, z])
            return (
                (1 - c) * v @ v.T
                + c * sp.eye(3)
                + s * Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            )

        def vRot(V, theta):
            return sp.diag(rot_vector(V, theta), 1)

        Mx = xRot(phi)
        Mz = zRot(theta)
        Mt = tLat([dx, dy, dz])
        column_major = False

        def ij_to_flat(i, j):
            return i + j * 4 if column_major else i * 4 + j

        def flat_to_ij(l):
            (i, j) = divmod(l, 4)
            return (j, i) if column_major else (i, j)

        def print_mMul(output=sys.stdout):
            print("export function mMul(")
            print(
                "    ",
                ",\n     ".join(
                    [
                        "["
                        + ", ".join(
                            [
                                f"{c}{i}{j}"
                                for l in range(16)
                                for (i, j) in (flat_to_ij(l),)
                            ]
                        )
                        + "]"
                        for c in "ba"
                    ]
                ),
                ") {\n  const C = new Float32Array(16);",
                file=output,
                sep="",
            )
            for l in range(16):
                (i, j) = flat_to_ij(l)
                print(
                    f"  C[{l}] =",
                    " + ".join([f"b{i}{k}*a{k}{j}" for k in range(4)]),
                    ";",
                    file=output,
                    sep="",
                )
            print("  return C;\n}", file=output, sep="")

        def write_ES6_module(output, column_major=False):
            def flatten(A):
                return [aij for aij in (A.T if column_major else A)]

            def ij_to_l(i, j):
                l = i + j * 4 if column_major else i * 4 + j
                return l

            if column_major:
                print("// Column major matrix functions", file=output, sep="")
            else:
                print("// Row major matrix functions", file=output, sep="")
            print(
                "const {sin,cos,sqrt,tan,PI}=Math,\n       pi=PI;\n",
                file=output,
                sep="",
            )
            for fRot in (xRot, yRot, zRot):
                print(
                    "export function "
                    + fRot.__name__
                    + "(a){\n  const s=sin(a);\n  const c=cos(a);\n  return ",
                    flatten(fRot(phi).subs({sin(phi): "s", cos(phi): "c"})),
                    ";\n}\n",
                    file=output,
                    sep="",
                )
            (x, y, z, theta, s, c, c1) = sp.symbols("x,y,z,theta,s,c,c1")
            print(
                'export function vRot([x,y,z],theta){\n  x??=0; y??=0; z??=0;\n  const length = sqrt(x*x + y*y + z*z);\n  if (length==0) {\n    if (theta===undefined){ \n       return [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];\n    }\n    else {\n       throw new Error("Rotation axis vector cannot be zero if a rotation angle is specified!");\n    }\n  }\n  if (theta===undefined) theta=length;\n  const c=cos(theta);\n  const c1=1-c;\n  const s=sin(theta);\n  x/=length;\n  y/=length;\n  z/=length;\n  return',
                flatten(
                    vRot(Matrix([x, y, z]), theta).subs(
                        {sin(theta): s, 1 - cos(theta): c1, -c1 + 1: c}
                    )
                ),
                ";\n}\n",
                file=output,
                sep="",
            )
            print(
                "export function tLat([tx,ty,tz]){\n  tx??=0; ty??=0; tz??=0;\n  return ",
                flatten(tLat(["tx", "ty", "tz"])),
                ";\n}\n",
                file=output,
                sep="",
            )
            print(
                "export function scal([sx, sy, sz]) {\n  sx??=1; sy??=1; sz??=1;\n  return ",
                flatten(scal(["sx", "sy", "sz"])),
                ";\n}\n",
                file=output,
                sep="",
            )
            T = [""] * 16
            for i in range(4):
                for j in range(4):
                    T[ij_to_l(i, j)] = f"A[{ij_to_l(j, i)}]"
            print(
                "export function T(A){\n  return new Float32Array([",
                ", ".join(T),
                "]);\n}\n",
                file=output,
                sep="",
            )
            print(
                "export function mMul(B,A){\n  const C=new Array(16);\n  let sum;\n  for (let i=0;i<4;++i)\n    for (let j=0;j<4;++j){\n      sum=0;\n      for (let k=0;k<4;++k)\n        sum+= B[",
                ij_to_l(*sp.symbols("i,k")),
                "] * A[",
                ij_to_l(*sp.symbols("k,j")),
                "];\n      C[",
                ij_to_l(*sp.symbols("i,j")),
                "] = sum;\n    }\n  return C;\n}\n",
                file=output,
                sep="",
            )
            print(
                "export function vMul(A,[x0,x1,x2,x3]){\n  x0??=0; x1??=0; x2??=0; x3??=0;\n  return new Float32Array([",
                ", ".join(
                    [
                        "+".join([f"A[{ij_to_l(i, k)}]*x{k}" for k in range(4)])
                        for i in range(4)
                    ]
                ),
                "]);\n}\n",
                file=output,
                sep="",
            )
            (fov, aspR, near, far, _f, nfInv) = sp.symbols(
                "fov,aspR,near,far,f,nfInv"
            )
            cot = sp.cot
            print(
                "export function  persp(fov, aspR, near, far) {\n  const f = tan(pi * 0.5 - 0.5 * fov);\n  const nfInv = 1.0 / (near - far);\n  return ",
                flatten(
                    persp(fov, aspR, near, far).subs(
                        {cot(fov / 2): _f, near - far: 1 / nfInv}
                    )
                ),
                ";\n}\n",
                file=output,
                sep="",
            )
            if column_major:
                print(
                    "export function cMaj(A){return A;}\nexport function rMaj(A){return T(A);}\n",
                    file=output,
                    sep="",
                )
            else:
                print(
                    "export function cMaj(A){return T(A);}\nexport function rMaj(A){return A;}\n",
                    file=output,
                    sep="",
                )

        def camMat(targ, azim, elev, d):
            return (
                tLat([0, 0, -d])
                @ xRot(elev - pi / 2)
                @ zRot(-azim - pi / 2)
                @ tLat(-sp.Matrix(targ))
            )

        def icamMat(targ, azim, elev, d):
            return (
                tLat(sp.Matrix(targ))
                @ zRot(-azim - pi / 2).T
                @ xRot(elev - pi / 2).T
                @ tLat([0, 0, d])
            )

        (a00, a01, a02, a10, a11, a12, a20, a21, a22) = sp.symbols(
            "a00,a01,a02,a10,a11,a12,a20,a21,a22"
        )

        def icamMat1(targ, camMat, d):
            return sp.Matrix(
                sp.BlockMatrix(
                    [
                        [camMat[:3, :3].T, d * camMat[2, :3].T + sp.Matrix(targ)],
                        [sp.zeros(1, 3), sp.ones(1, 1)],
                    ]
                )
            )

        def print_camMat(output=sys.stdout, column_major=True):
            (tx, ty, tz, azim, elev, d) = sp.symbols("tx,ty,tz,azim,elev,d")

            def flatten(A):
                return [aij for aij in (A.T if column_major else A)]

            print(
                "export function camMat([tx,ty,tz],azim,elev,d){\n  // The function camMat calculates the camera matrix (similar to lookAt, but with different input parameters)\n  // tx,ty,tz: target coordinates\n  // azim: azimuth angle in radians\n  // elev: elevation angle in radians\n  // d: distance of camera from target. \n  tx??=0; ty??=0; tz??=0; d??=0;\n  const s=sin(azim),\n        c=cos(azim),\n        se=sin(elev),\n        ce=cos(elev);\n  return new Float32Array(",
                flatten(
                    camMat([tx, ty, tz], azim, elev, d).subs(
                        {
                            sin(elev): "se",
                            cos(elev): "ce",
                            sin(azim): "s",
                            cos(azim): "c",
                        }
                    )
                ),
                ")\n};\n",
                file=output,
                sep="",
            )

        def camPos(targ, camMat, dist):
            ex = camMat[2, 0]
            ey = camMat[2, 1]
            ez = camMat[2, 2]
            (tx, ty, tz, *_) = targ
            return [tx + ex * dist, ty + ey * dist, tz + ez * dist, 1]

        def print_camPos(output=sys.stdout, column_major=True):
            def ij_to_l(i, j):
                l = i + j * 4 if column_major else i * 4 + j
                return l

            print(
                "export function camPos(targ,camMat,d){\n  //Camera position in world coordinates  // tx,ty,tz: target coordinates\n  // camMat: camera matrix\n  // d: distance of camera from target. \n  const [tx,ty,tz]=targ;\n  const ex=camMat[",
                ij_to_l(2, 0),
                "], ey=camMat[",
                ij_to_l(2, 1),
                "], ez=camMat[",
                ij_to_l(2, 2),
                "];\n  return [tx+ex*d,ty+ey*d,tz+ez*d,1];\n};\n",
                file=output,
                sep="",
            )

        def print_icamMat(output=sys.stdout, column_major=True):
            def ij_to_l(i, j):
                l = i + j * 4 if column_major else i * 4 + j
                return l

            iC = ["0"] * 16
            iC[15] = "1"
            for i in range(3):
                for j in range(3):
                    iC[ij_to_l(i, j)] = f"C[{ij_to_l(j, i)}]"
                iC[ij_to_l(i, 3)] = f"C[{ij_to_l(2, i)}]*d+t[{i}]??0"
            print(
                "export function icamMat(t,C,d){\n  // The function icamMat calculates the inverse of the camera matrix for a given camera matrix\n  // t: target coordinates\n  // C: camera matrix\n  // d: distance of camera from target. \n  d??=0;\n  return new Float32Array([",
                ", ".join(iC),
                "])\n};\n",
                file=output,
                sep="",
            )

        with vfs.open("m4_cMaj.js", "w") as _f:
            write_ES6_module(_f, column_major=True)
            print_camMat(_f, column_major=True)
            print_icamMat(_f, column_major=True)
            print_camPos(_f, column_major=True)
        m4_cMaj_js = vfs["m4_cMaj.js"]
        with vfs.open("m4_rMaj.js", "w") as _f:
            write_ES6_module(_f, column_major=False)
            print_camMat(_f, column_major=False)
            print_icamMat(_f, column_major=False)
            print_camPos(_f, column_major=False)
        m4_rMaj_js = vfs["m4_rMaj.js"]
        camera = dict(
            fov=30 * deg, target=[0, 0, 0], azim=30 * deg, elev=40 * deg, dist=1000
        )
        a = {
            0: 0.8660253882408142,
            1: -0.3830222189426422,
            10: -0.7660444378852844,
            11: 0,
            12: 0,
            13: 0,
            14: -1000,
            15: 1,
            2: -0.3213938176631927,
            3: 0,
            4: -0.5,
            5: -0.663413941860199,
            6: -0.5566704273223877,
            7: 0,
            8: 0,
            9: 0.6427876353263855,
        }
        b = [
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
            1,
        ]
        (tx, ty, tz, azim, elev, d) = sp.symbols("tx,ty,tz,azim,elev,d")
        cm = camMat([tx, ty, tz], azim, elev, d).subs(
            {sin(elev): "se", cos(elev): "ce", sin(azim): "s", cos(azim): "c"}
        )
        icm = icamMat([tx, ty, tz], azim, elev, d).subs(
            {sin(elev): "se", cos(elev): "ce", sin(azim): "s", cos(azim): "c"}
        )
        print("camera matrix:")
        sp.pprint(cm)
        print()
        print("inverse camera matrix:")
        sp.pprint(icm)
        print()
        print("icamMat1([tx,ty,tz],cm,d):")
        sp.pprint(icamMat1([tx, ty, tz], cm, d))
        cm = sp.Matrix([[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]])
        print()
        print("icamMat1([tx,ty,tz],aij,d):")
        sp.pprint(icamMat1([tx, ty, tz], cm, d))
        (tx, ty, tz, azim, elev, d) = (
            0,
            0,
            0,
            camera["azim"],
            camera["elev"],
            camera["dist"],
        )
        cm = camMat([tx, ty, tz], azim, elev, d)
        icm = icamMat([tx, ty, tz], azim, elev, d)
        print()
        print("(cm.inv() @s p.Matrix([0,0,0,1]) ).evalf():")
        sp.pprint((cm.inv() @ sp.Matrix([0, 0, 0, 1])).evalf())
        print()
        print("camPos([tx,ty,tz],cm.evalf(),d):")
        print(camPos([tx, ty, tz], cm.evalf(), d))
        print()
        print("check that (cm @ icm).evalf()  = I :")
        sp.pprint((cm @ icm).evalf())
        return m4_cMaj_js,m4_rMaj_js
    m4_cMaj_js,m4_rMaj_js=_()
    vfs["m4_cMaj.js"],vfs["m4_rMaj.js"]=m4_cMaj_js,m4_rMaj_js
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## HTML & JavaScript files""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### webgl-torus.html""")
    return


@app.cell
def _(vfs):
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
    webgl_torus_html=vfs["webgl-torus.html"]
    return (webgl_torus_html,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### webgl-torus.js""")
    return


@app.cell
def _(vfs):
    webgl_torus_js=r"""
    let elementIsDefined
    if (typeof element === 'undefined') {
       elementIsDefined=false;
       window.element=document.body;
    }
    else{
       elementIsDefined=true;
    }
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

    const wrapper = document.createElement("div");
    wrapper.style.position = "relative";
    wrapper.style.display = "inline-block";

    export const hint = document.createElement("div");
    hint.innerText = "Colors indicate surface normal directions: +xyz=rgb, -xyz=cmy \nDrag to rotate the view.";

    if (typeof logdiv==='undefined'){ hint.innerText+="\nelementIsDefined="+elementIsDefined;
    }
    else{
    logdiv.innerText+="\nelementIsDefined="+elementIsDefined;
    }

    wrapper.appendChild(hint);
    if (!(typeof element.logdiv==='undefined')) {
    element.logdiv.innerText+="initializing";
    }

    export const canvas = document.createElement("canvas");
    canvas.style.display = "block";
    canvas.style.width = "400px";
    canvas.style.height = "400px";
    wrapper.appendChild(canvas);

    element.appendChild(wrapper);
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
    export const style={hideButton:false};

    import { cube, circle, extrude } from './geometry.js';
    import { cookiecutters } from './cookiecutters.js';
    import { Segments2Complex } from './turtle-graphics.js';

    export const ShapeData={};


    function getOutlinePath(name,{scale=1.0}){
    const { turtlePath, startPoint, startAngle } = cookiecutters.outlines[name];
      const p0 = startPoint || [0, 0];
      const a0 = [Math.cos(startAngle * deg), Math.sin(startAngle * deg)];
      const segs = turtlePath.map(([l, a]) => [l, a * deg]);
      const S2C= Segments2Complex({ 
          segs:segs,
      p0_a0_segs: [[[p0[0]*scale,p0[1]*scale], a0], segs],
        scale: scale, // Adjust to match radius 150 (circumference ~62.83185)
        tol: 0.05,
        loops: 1,
        return_start: false
      });
      const pathPoints = Array.from(S2C);
      const epath = pathPoints.map(({ point, angle }) => [point, angle]);
    //  console.log(epath);

      if (name==='Circle'){
      return circle(scale*10,Math.round(13*Math.sqrt(scale)));  
      }

      return epath;
    }


    function initialize() {
        let scale, epath, spath;
    //  epath=getOutlinePath("Plain",{scale:15});    
    //  spath=getOutlinePath("Plain",{scale:0.5});
      epath=getOutlinePath("Duck",{scale:11});
     spath=getOutlinePath("Blade",{scale:5});
       Object.assign(ShapeData, extrude(epath, spath));
      // ... rest of initialize remains unchanged
    //}

    //  Object.assign(ShapeData,(extrude(circle(150,50),circle(50.0,30))));
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

    // Add animation state function
    export function setAnimationState(animate) {
        // Start animation if not already animating
        if (animate && !state.animate) { 
            requestAnimationFrame(drawScene);
        }
        state.animate = animate;        
        animBtn.style.display = state.animate || style.hideButton ? 'none' : 'block';
    }

    // Add button with overlay
    export const animBtn = document.createElement('button');
    animBtn.id = 'animBtn';
    animBtn.className = '';
    animBtn.className = '';
    animBtn.style.padding = '5px 10px';
    animBtn.style.backgroundColor = 'var(--slate-1, #f0f0f0)'; // Light gray
    animBtn.style.border = '1px solid var(--slate-6, #ccc)';
    animBtn.style.color = 'var(--slate-11, #333)';
    animBtn.style.borderRadius = '4px';
    animBtn.style.cursor = 'pointer';
    animBtn.textContent = 'Start Animation';
    animBtn.style.position = 'absolute';
    animBtn.style.bottom = '10px';
    animBtn.style.left = '10px';
    //animBtn.style.padding = '5px 10px';
    animBtn.style.zIndex = '10';
    //animBtn.style.backgroundColor = 'buttonface';
    //animBtn.style.border = '2px outset buttonborder';
    //animBtn.style.cursor = 'pointer';
    animBtn.addEventListener('click', () => setAnimationState(true));
    wrapper.appendChild(animBtn);
    setAnimationState(state.animate); // Show/hide button according to initial state

    // Stop animation on drag
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        if (state.animate) {
            setAnimationState(false);
        }
    });
    canvas.addEventListener('mousedown', (e) => {
        if (state.animate) {
            setAnimationState(false);
        }
    });

    import * as M4 from './m4_cMaj.js';
    initialize();
    """
    vfs["webgl-torus.js"]=webgl_torus_js
    return (webgl_torus_js,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### interaction.js""")
    return


@app.cell
def _(vfs):
    interaction_js = r"""
    // TouchHandler: Handles single-touch/mouse for TrackpadWidget
    export class TouchHandler {
      constructor(c, cb) {
        if (!c || !(c instanceof HTMLCanvasElement)) {
          console.error('TouchHandler: Invalid canvas', c);
          throw new Error('Invalid canvas');
        }
        this.c = c;
        this.cb = cb || {};
        this.d = false;
        this.x = 0;
        this.y = 0;
        this.init();
      }

      init() {
        console.log('TouchHandler: Initializing on canvas', this.c);
        const evts = [
          ['touchstart', e => this.hS(e)],
          ['touchmove', e => this.hM(e)],
          ['touchend', e => this.hE(e)],
          ['mousedown', e => this.hMD(e)],
          ['mousemove', e => this.hMM(e)],
          ['mouseup', e => this.hMU(e)]
        ];
        evts.forEach(([t, f]) => {
          this.c.addEventListener(t, f.bind(this));
          console.log('TouchHandler: Bound event', t);
        });
        document.addEventListener('touchmove', e => {
          if (e.target === this.c) e.preventDefault();
        }, { passive: false });
      }

      pos(e) {
        const r = this.c.getBoundingClientRect();
        const t = e.touches?.[0] || e.changedTouches?.[0] || e;
        const x = t.clientX - r.left;
        const y = t.clientY - r.top;
        console.log('TouchHandler: pos', { clientX: t.clientX, left: r.left, x, clientY: t.clientY, top: r.top, y });
        return { x: x, y: y };
      }

      hS(e) {
        e.preventDefault();
        this.d = true;
        const { x, y } = this.pos(e);
        this.x = x;
        this.y = y;
        console.log('TouchHandler: touchstart', { x, y });
        if (this.cb.start) this.cb.start(x, y);
      }

      hM(e) {
        e.preventDefault();
        if (!this.d) return;
        const { x, y } = this.pos(e);
        const dx = x - this.x;
        const dy = y - this.y;
        console.log('TouchHandler: touchmove', { x, y, dx, dy });
        if (this.cb.move) this.cb.move(dx, dy, x, y);
        this.x = x;
        this.y = y;
      }

      hE(e) {
        e.preventDefault();
        this.d = false;
        console.log('TouchHandler: touchend');
        if (this.cb.end) this.cb.end();
      }

      hMD(e) {
        this.d = true;
        const { x, y } = this.pos(e);
        this.x = x;
        this.y = y;
        console.log('TouchHandler: mousedown', { x, y });
        if (this.cb.start) this.cb.start(x, y);
      }

      hMM(e) {
        if (!this.d) return;
        const { x, y } = this.pos(e);
        const dx = x - this.x;
        const dy = y - this.y;
        console.log('TouchHandler: mousemove', { x, y, dx, dy });
        if (this.cb.move) this.cb.move(dx, dy, x, y);
        this.x = x;
        this.y = y;
      }

      hMU(e) {
        this.d = false;
        console.log('TouchHandler: mouseup');
        if (this.cb.end) this.cb.end();
      }
    }
    """
    vfs["interaction.js"]=interaction_js
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### m4_cMaj.js
    (code generated by the Python script above)
    """
    )
    return


@app.cell
def _(vfs):
    vfs["m4_cMaj.js"]="""
    // Column major matrix functions
    const {sin,cos,sqrt,tan,PI}=Math,
           pi=PI;

    export function xRot(a){
      const s=sin(a);
      const c=cos(a);
      return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1];
    }

    export function yRot(a){
      const s=sin(a);
      const c=cos(a);
      return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1];
    }

    export function zRot(a){
      const s=sin(a);
      const c=cos(a);
      return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    }

    export function vRot([x,y,z],theta){
      x??=0; y??=0; z??=0;
      const length = sqrt(x*x + y*y + z*z);
      if (length==0) {
        if (theta===undefined){ 
           return [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
        }
        else {
           throw new Error("Rotation axis vector cannot be zero if a rotation angle is specified!");
        }
      }
      if (theta===undefined) theta=length;
      const c=cos(theta);
      const c1=1-c;
      const s=sin(theta);
      x/=length;
      y/=length;
      z/=length;
      return[c + c1*x**2, c1*x*y + s*z, c1*x*z - s*y, 0, c1*x*y - s*z, c + c1*y**2, c1*y*z + s*x, 0, c1*x*z + s*y, c1*y*z - s*x, c + c1*z**2, 0, 0, 0, 0, 1];
    }

    export function tLat([tx,ty,tz]){
      tx??=0; ty??=0; tz??=0;
      return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, tx, ty, tz, 1];
    }

    export function scal([sx, sy, sz]) {
      sx??=1; sy??=1; sz??=1;
      return [sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, 0, 0, 0, 1];
    }

    export function T(A){
      return new Float32Array([A[0], A[4], A[8], A[12], A[1], A[5], A[9], A[13], A[2], A[6], A[10], A[14], A[3], A[7], A[11], A[15]]);
    }

    export function mMul(B,A){
      const C=new Array(16);
      let sum;
      for (let i=0;i<4;++i)
        for (let j=0;j<4;++j){
          sum=0;
          for (let k=0;k<4;++k)
            sum+= B[i + 4*k] * A[4*j + k];
          C[i + 4*j] = sum;
        }
      return C;
    }

    export function vMul(A,[x0,x1,x2,x3]){
      x0??=0; x1??=0; x2??=0; x3??=0;
      return new Float32Array([A[0]*x0+A[4]*x1+A[8]*x2+A[12]*x3, A[1]*x0+A[5]*x1+A[9]*x2+A[13]*x3, A[2]*x0+A[6]*x1+A[10]*x2+A[14]*x3, A[3]*x0+A[7]*x1+A[11]*x2+A[15]*x3]);
    }

    export function  persp(fov, aspR, near, far) {
      const f = tan(pi * 0.5 - 0.5 * fov);
      const nfInv = 1.0 / (near - far);
      return [f/aspR, 0, 0, 0, 0, f, 0, 0, 0, 0, nfInv*(far + near), -1, 0, 0, 2*far*near*nfInv, 0];
    }

    export function cMaj(A){return A;}
    export function rMaj(A){return T(A);}

    export function camMat([tx,ty,tz],azim,elev,d){
      // The function camMat calculates the camera matrix (similar to lookAt, but with different input parameters)
      // tx,ty,tz: target coordinates
      // azim: azimuth angle in radians
      // elev: elevation angle in radians
      // d: distance of camera from target. 
      tx??=0; ty??=0; tz??=0; d??=0;
      const s=sin(azim),
            c=cos(azim),
            se=sin(elev),
            ce=cos(elev);
      return new Float32Array([-s, -c*se, c*ce, 0, c, -s*se, ce*s, 0, 0, ce, se, 0, -c*ty + s*tx, c*se*tx - ce*tz + s*se*ty, -c*ce*tx - ce*s*ty - d - se*tz, 1])
    };

    export function icamMat(t,C,d){
      // The function icamMat calculates the inverse of the camera matrix for a given camera matrix
      // t: target coordinates
      // C: camera matrix
      // d: distance of camera from target. 
      d??=0;
      return new Float32Array([C[0], C[4], C[8], 0, C[1], C[5], C[9], 0, C[2], C[6], C[10], 0, C[2]*d+t[0]??0, C[6]*d+t[1]??0, C[10]*d+t[2]??0, 1])
    };

    export function camPos(targ,camMat,d){
      //camera position in world coordinates  // tx,ty,tz: target coordinates
      // camMat: camera matrix
      // d: distance of camera from target. 
      const [tx,ty,tz]=targ;
      const ex=camMat[2], ey=camMat[6], ez=camMat[10];
      return [tx+ex*d,ty+ey*d,tz+ez*d,1];
    };

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### geometry.js""")
    return


@app.cell
def _(vfs):
    geometry_js=r"""
    // geometry.js
    export function cube(dx=1, dy, dz) {
      const numVertices = 24;
      const stride = 6 * 4;
      const X = 0.5 * dx;
      const Y = dy | X;
      const Z = dz | X;
      const vertices = new Float32Array([
        X,-Y,-Z, 1, 0, 0, X, Y,-Z, 1, 0, 0, X, Y, Z, 1, 0, 0, X,-Y, Z, 1, 0, 0,
        -X, Y, Z,-1, 0, 0,-X, Y,-Z,-1, 0, 0,-X,-Y,-Z,-1, 0, 0,-X,-Y, Z,-1, 0, 0,
        -X, Y,-Z, 0, 1, 0,-X, Y, Z, 0, 1, 0, X, Y, Z, 0, 1, 0, X, Y,-Z, 0, 1, 0,
        X,-Y, Z, 0,-1, 0,-X,-Y, Z, 0,-1, 0,-X,-Y,-Z, 0,-1, 0, X,-Y,-Z, 0,-1, 0,
        -X,-Y, Z, 0, 0, 1, X,-Y, Z, 0, 0, 1, X, Y, Z, 0, 0, 1,-X, Y, Z, 0, 0, 1,
        X, Y,-Z, 0, 0,-1, X,-Y,-Z, 0, 0,-1,-X,-Y,-Z, 0, 0,-1,-X, Y,-Z, 0, 0,-1,
      ]);
      const indices = new Int16Array([
        0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9,10,10,11, 8,
        12,13,14,14,15,12, 16,17,18,18,19,16, 20,21,22,22,23,20
      ]);
      return { indices, vertices, stride, numVertices };
    }

    export function circle(r, n) {
      const epath = [];
      for (let i = 0; i < n; i++) {
        const theta = i * 2 * Math.PI / n;
        const s = Math.sin(theta);
        const c = Math.cos(theta);
        epath[i] = [[r * c, r * s], [-s, c]];
      }
      return epath;
    }

    export function extrude(epath, shape) {
      const m = epath.length;
      const n = shape.length;
      const numVertices = m * n;
      const vertices = new Float32Array(numVertices * 6);
      const stride = 6 * 4;
      const indices = new Int16Array(numVertices * 6);
      for (let j = 0; j < m; j++) {
        let [[x_p, y_p], [ms_p, c_p]] = epath[j];
        for (let i = 0; i < n; i++) {
          const [[x_s, y_s], [ms_s, c_s]] = shape[i];
          const k = j * n + i;
          vertices[k * 6 + 0] = x_p + x_s * c_p;
          vertices[k * 6 + 1] = y_p - x_s * ms_p;
          vertices[k * 6 + 2] = y_s;
          vertices[k * 6 + 3] = c_s * c_p;
          vertices[k * 6 + 4] = -c_s * ms_p;
          vertices[k * 6 + 5] = -ms_s;
          indices[k * 6 + 0] = j * n + i;
          indices[k * 6 + 1] = ((j + 1) % m) * n + ((i + 1) % n);
          indices[k * 6 + 2] = j * n + ((i + 1) % n);
          indices[k * 6 + 3] = j * n + i;
          indices[k * 6 + 4] = ((j + 1) % m) * n + i;
          indices[k * 6 + 5] = ((j + 1) % m) * n + ((i + 1) % n);
        }
      }
      return { indices, vertices, stride, numVertices };
    }"""
    vfs["geometry.js"]=geometry_js
    return (geometry_js,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### turtle-graphics.js""")
    return


@app.cell
def _(vfs):
    turtle_graphics_js=r"""
    export function TurtlePathLengthArea(TurtlePath,arcStartAngle=0) {
        /**
         * Calculates the length, area, end point, final angle, and centroid of a shape formed by arc segments.
         * 
         * @param {Array} TurtlePath - An array of arrays, where each inner array contains [arc_length, arc_angle].
         * @return {Array} An array containing:
         *   - Total length
         *   - Total area
         *   - End point of the last arc as an array [x, y]
         *   - Final angle after all rotations in radians
         *   - Centroid [x, y] of the shape
         */
        let totalLength = 0; // Total length
        let totalArea = 0; // Total area
        let firstMoment = [0, 0]; // 1st moment around x and y axis
        let arcStartPoint = [0, 0]; // Starting point of each arc segment
    //    let arcStartAngle = 0; // Starting angle of each arc
        let arcEndAngle = 0; // Ending angle of each arc
        for (let [arcLength, arcAngle] of TurtlePath) {
            // Pre-calculate frequently used terms:
            const halfArcAngle = arcAngle / 2;
            const chordAngle = arcStartAngle + halfArcAngle;
    		const cosChordAngle = Math.cos(chordAngle);
            const sinChordAngle = Math.sin(chordAngle);


            // Calculate length
            totalLength += arcLength;
    		let chordLength;
            if (arcAngle !== 0) {
                if (arcLength !== 0) { // No need to update areas for sharp turns
                    const radius = arcLength / arcAngle;
                    const sinHalfArcAngle = Math.sin(halfArcAngle);
                    chordLength	= radius * sinHalfArcAngle * 2; // Chord length
                    const arcSegmentArea = 0.5 * radius ** 2 * (arcAngle - Math.sin(arcAngle)); // Arc segment's area
                    const y_a = (2/3) * (radius * sinHalfArcAngle) ** 3;
                    totalArea += arcSegmentArea;
                    firstMoment[0] += arcSegmentArea * (arcStartPoint[0] - radius * Math.sin(arcStartAngle)) + y_a * sinChordAngle;
                    firstMoment[1] += arcSegmentArea * (arcStartPoint[1] + radius * Math.cos(arcStartAngle)) - y_a * cosChordAngle;
                } else {
                    chordLength = 0;
                }
            } else {
                chordLength = arcLength;
            }

            // Calculate the end point of this arc segment
            const arcEndPoint = [
                arcStartPoint[0] + chordLength * cosChordAngle,
                arcStartPoint[1] + chordLength * sinChordAngle
            ];
            arcEndAngle = arcStartAngle + arcAngle;

            // Shoelace formula for area
            const triangleArea = (arcStartPoint[0] * arcEndPoint[1] - arcEndPoint[0] * arcStartPoint[1]) / 2;
            totalArea += triangleArea;
            firstMoment[0] += triangleArea * (arcStartPoint[0] + arcEndPoint[0]) / 3;
            firstMoment[1] += triangleArea * (arcStartPoint[1] + arcEndPoint[1]) / 3;

            // Update for next iteration
            arcStartPoint = arcEndPoint;
            arcStartAngle = arcEndAngle;
        }

        const centroid = [firstMoment[0] / totalArea, firstMoment[1] / totalArea];
        return [totalLength, totalArea, arcStartPoint, arcEndAngle, centroid];
    }


    export function* Segments2Complex({ p0_a0_segs = [[[0, 0], [1, 0]], []], scale = 1.0, tol = 0.05, offs = 0, loops = 1, return_start = false }) {
    //    console.log('in Segments2Complex');
        const [p0, a0] = p0_a0_segs[0];
        const Segs = p0_a0_segs[1];
        let a = a0.slice();
        let p = p0.slice();
        p[0] = p[0] - a[0] * offs; // Real part
        p[1] = p[1] - a[1] * offs; // Imaginary part
        let L = 0;

        if (return_start) {
            yield { point: p, angle: a, length: L, segmentIndex: -1 };
        }

        let loopcount = 0;
        while (loops === null || loops === Infinity || loopcount < loops) {
            loopcount++;
            for (let X = 0; X < Segs.length; X++) {
                let [l, da, ..._] = Segs[X];
                l *= scale;
                let n;
    			let v;
    			let dda;
                if (da !== 0) {  // If da is not zero
                    let r = l / da;
                    r += offs;
                    if (r !== 0) {
                        l = r * da;
                        let dl = 2 * Math.sqrt(2 * Math.abs(r) * tol);
                        n = Math.max(Math.ceil(6 * Math.abs(da / (2 * Math.PI))), Math.floor(l / dl) + 1);
                        let dda2 = [Math.cos(0.5*da / n), Math.sin(0.5*da / n)];
                        v = [2 * r* dda2[1] * dda2[0], 2 * r* dda2[1] * dda2[1]];
                        v = [v[0] * a[0] - v[1] * a[1], v[0] * a[1] + v[1] * a[0]];
                    } else {
                        n = 1;
                        v = [0,0];
                    }
    				dda = [Math.cos(da / n), Math.sin(da / n)];
                    for (let i = 0; i < n; i++) {
                        L += l / n;
                        p[0] += v[0];
                        p[1] += v[1];
    					a = [a[0] * dda[0] - a[1] * dda[1], a[0] * dda[1] + a[1] * dda[0]];
                        yield { point: p.slice(), angle: a, length: L, segmentIndex: X };
                        v = [v[0] * dda[0] - v[1] * dda[1], v[0] * dda[1] + v[1] * dda[0]];
                    }
                } else {
                    n = 1; // Set n to 1 for the zero case
                    // Handle the case when da is zero
                    L += l;
                    p[0] += l * a[0];
                    p[1] += l * a[1];
                    yield { point: p.slice(), angle: a, length: L, segmentIndex: X };
                }
            }
        }
    }

    // Usage example:

    export function plot_segments(ctx,{p0=[0,0],a0=[1,0],segs=[],scale=1.0,tol=0.05,offs=0,loops=1,return_start=true}={}){
    //	debugLog("segs: "+segs);
    	let gen = Segments2Complex({
        p0_a0_segs: [[p0, a0],segs],
        scale: scale,
        tol: tol,
        offs: 0,
        loops: loops,
        return_start: return_start
    });
    //    segs.forEach((x,i)=>{debugLog(i+x);});
        ctx.beginPath();
        let {value:{point,angle:[cos_ang,sin_ang]}}=gen.next();
    //	let point=value.point;
    //	debugLog("point: "+point);
        ctx.moveTo(point[0]-offs*sin_ang,point[1]+offs*cos_ang);
    	for (let {point,angle:[cos_ang,sin_ang]} of gen){
    //		debugLog("point: "+point);
    		ctx.lineTo(point[0]-offs*sin_ang,point[1]+offs*cos_ang);
    	}
    	ctx.stroke()
    }
    """
    vfs['turtle-graphics.js']=turtle_graphics_js
    return (turtle_graphics_js,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### cookiecutters.json""")
    return


@app.cell
def _(vfs):
    cookiecutters_json="""
    {
      "outlines": {
        "Star": { 
    	  "turtlePath":[
          [ 2, -58.0 ],
          [ 8, 0.0 ],
          [ 3.2, 130.0 ],
          [ 8, 0.0 ],
          [ 2, -58.0 ],
          [ 8, 0.0 ],
          [ 3.2, 130.0 ],
          [ 8, 0.0 ],
          [ 2, -58.0 ],
          [ 8, 0.0 ],
          [ 3.2, 130.0 ],
          [ 8, 0.0 ],
          [ 2, -58.0 ],
          [ 8, 0.0 ],
          [ 3.2, 130.0 ],
          [ 8, 0.0 ],
          [ 2, -58.0 ],
          [ 8, 0.0 ],
          [ 3.2, 130.0 ],
          [ 8, 0.0 ]
        ]},
        "Plain":{ 
          "startPoint":[10, 0.0],
    	  "startAngle": 90.0,
    	  "turtlePath": [
          [ 62.83185, 360.0 ]
        ]},
        "Scalloped":{ 
    	  "turtlePath": [
          [ 1.0, -110.0 ],
          [ 2, 150.0 ],
          [ 1.0, -110.0 ],
          [ 2, 150.0 ],
          [ 1.0, -110.0 ],
          [ 2, 150.0 ],
          [ 1.0, -110.0 ],
          [ 2, 150.0 ],
          [ 1.0, -110.0 ],
          [ 2, 150.0 ],
          [ 1.0, -110.0 ],
          [ 2, 150.0 ],
          [ 1.0, -110.0 ],
          [ 2, 150.0 ],
          [ 1.0, -110.0 ],
          [ 2, 150.0 ],
          [ 1.0, -110.0 ],
          [ 2, 150.0 ]
        ]},
        "Heart":{
    	  "startAngle": 180,
    	  "turtlePath": [
          [ 0.45, -45.0 ],
          [ 10.0, 180.0 ],
          [ 6.91, -10.0 ],
          [ 1.1, 110.0 ],
          [ 6.91, -10.0 ],
          [ 10.0, 180.0 ],
          [ 0.45, -45.0 ]
        ]},
        "Duck":{ 
    	  "startAngle": 180, 
    	  "turtlePath": [
          [ 0.4, -10.0 ],
          [ 13.297, 25.0 ],
          [ 3, -80.0 ],
          [ 4, 160.0 ],
          [ 22.913, 90.0 ],
          [ 15, 90.0 ],
          [ 5, -90.0 ],
          [ 5, 20.0 ],
          [ 3, 170.0 ],
          [ 2, -20.0 ],
          [ 3, -90.0 ],
          [ 15, 220.0 ],
          [ 5, -125.0 ]
        ]},
        "Tree":{ 
    	  "startAngle":180,
    	  "turtlePath": [
          [ 0.75, 40.0 ],
          [ 3, 0.0 ],
          [ 1.5, 140.0 ],
          [ 0.6, 0.0 ],
          [ 1.0, -140.0 ],
          [ 3, 0.0 ],
          [ 1.5, 140.0 ],
          [ 0.6, 0.0 ],
          [ 1.0, -140.0 ],
          [ 3, 0.0 ],
          [ 1.5, 140.0 ],
          [ 0.6, 0.0 ],
          [ 1.0, -140.0 ],
          [ 3, 0.0 ],
          [ 1.5, 140.0 ],
          [ 11.431, 0.0 ],
          [ 1.5, 140.0 ],
          [ 3, 0.0 ],
          [ 1.0, -140.0 ],
          [ 0.6, 0.0 ],
          [ 1.5, 140.0 ],
          [ 3, 0.0 ],
          [ 1.0, -140.0 ],
          [ 0.6, 0.0 ],
          [ 1.5, 140.0 ],
          [ 3, 0.0 ],
          [ 1.0, -140.0 ],
          [ 0.6, 0.0 ],
          [ 1.5, 140.0 ],
          [ 3, 0.0 ],
          [ 0.75, 40.0 ]
        ]}, 
    	"Blade":{ 
    	  "startPoint":[-1.8, 0.0],
    	  "startAngle": 0.0,
    	  "turtlePath":[
          [ 3.6, 0 ],
    	  [0,45],
    	  [0.661522368915,0],
          [ 3, 90],
    	  [ 2.5, 0],
          [0,-43.5679],
          [ 10, 0 ],
          [0,88.5679],
          [ 0.5, 0 ],
          [0,88.5679],
          [ 10, 0],
          [0,-43.5679],
    	  [ 2.5, 0],
          [ 3,  90],
    	  [0.661522368915,0],
    	  [0,45]
        ]},
    	"L":{ 
    	  "turtlePath": [
    	  [  8,  0 ],
    	  [  0, 90 ],
    	  [  4,  0 ],
    	  [  0, 90 ],
    	  [  8,  0 ],
    	  [  0,-180 ],
    	  [  8,  0 ],
    	  [  0, 90 ],
          [  4,  0 ],
    	  [  0, 90 ],
    	  [  8,  0 ],
    	  [  0, 90 ],
          [  8,  0 ],
    	  [  0,-90 ],
    	  [ 16,  0 ],
    	  [  0, 90 ],
          [  8,  0 ],
    	  [  0, 90 ],
          [  4,  0 ],
    	  [  0, 90 ],
    	  [  8,  0 ],
    	  [  0,-180 ],
          [  8,  0 ],
    	  [  0, 90 ],
          [  4,  0 ],
    	  [  0, 90 ],
    	  [  8,  0 ],
    	  [  0,-180 ],
          [  8,  0 ],
    	  [  0, 90 ],
          [  4,  0 ],
    	  [  0, 90 ],
    	  [  8,  0 ],
    	  [  0,-180 ],
          [  8,  0 ],
    	  [  0, 90 ],
          [  4,  0 ],
    	  [  0, 90 ],
    	  [  8,  0 ],
    	  [  0,-180 ],
          [  8,  0 ],
    	  [  0, 90 ],
    	  [  4,  0 ],
    	  [  0, 90 ],
    	  [ 16,  0 ],
    	  [  0,-180 ],
    	  [ 16,  0 ],
    	  [  0, 90 ],
    	  [  4,  0 ],
    	  [  0, 90 ],
    	  [  4,  0 ],
    	  [  0, 90 ],
    	  [ 24,  0 ],
    	  [  0,-180 ],
    	  [ 24,  0 ],
    	  [  0, 90 ],
    	  [  4,  0 ],
    	  [  0, 90 ],
    	  [  8,  0 ],
    	  [  0,-180 ]
        ]},
    	"A":{ 
    	  "turtlePath": [
    	  [  0,  70 ],
    	  [  0.4, 0 ],
    	  [  0,  -70 ],
    	  [  0.41, 0 ],
    	  [  0,  180 ],
    	  [  0.41, 0 ],
    	  [  0, -110 ],
    	  [  0.6, 0  ],
          [  0, -140 ],
    	  [  1, 0 ],
    	  [  0,  70 ]
        ]}, 
    	"B":{ 
    	  "turtlePath": [
    	  [  0,  90 ],
    	  [  1, 0 ],
    	  [  0,  -90 ],
    	  [  0.25, 0 ],
    	  [  0.684, -180 ],
    	  [  0.25,0 ],
    	  [  0, 180 ],
    	  [  0.25, 0 ],
    	  [  0.896, -180 ],
    	  [  0.25,0 ],
    	  [  0, 180 ]
        ]}
      },
      "brickworks": {
        "centered": [
          [ [ -0.45, 0.9 ], [ -1.35, 0.9 ], [ 0.45, 0.9 ], [ 1.35, 0.9 ] ],
          [ [ -0.8, 0.8 ], [ -1.6, 0.8 ], [ 0.8, 0.8 ], [ 1.6, 0.8 ], [ 0.0, 0.8 ] ],
          [ [ -1.1, 0.733 ], [ -1.833, 0.733 ], [ 1.1, 0.733 ], [ 1.833, 0.733 ], [ -0.367, 0.733 ], [ 0.367, 0.733 ] ],
          [ [ -0.92, 0.92 ], [ -1.84, 0.92 ], [ 0.92, 0.92 ], [ 1.84, 0.92 ], [ 0.0, 0.92 ] ],
          [ [ 0.0, 1.0 ], [ -1.9, 1.0 ], [ 1.9, 1.0 ] ],
          [ [ 0.0, 0.8 ], [ -2.0, 1.0 ], [ 2.0, 1.0 ] ],
          [ [ 0.0, 0.6 ], [ -2.05, 0.9 ], [ 2.05, 0.9 ] ],
          [ [ 0.0, 0.5 ], [ -2.05, 0.9 ], [ 2.05, 0.9 ] ],
          [ [ 0.0, 0.5 ], [ -2.05, 0.9 ], [ 2.05, 0.9 ] ],
          [ [ 0.0, 0.5 ], [ -1.95, 0.9 ], [ 1.95, 0.9 ] ],
          [ [ 0.0, 0.5 ], [ -1.85, 0.9 ], [ 1.85, 0.9 ] ],
          [ [ 0.0, 0.5 ], [ -1.7, 1.0 ], [ 1.7, 1.0 ] ],
          [ [ 0.0, 0.5 ], [ -1.5, 1.0 ], [ 1.5, 1.0 ] ],
          [ [ 0.0, 0.5 ], [ -1.3, 1.0 ], [ 1.3, 1.0 ] ],
          [ [ 0.0, 0.5 ], [ -1.1, 1.0 ], [ 1.1, 1.0 ] ],
          [ [ 0.0, 0.5 ], [ -0.9, 1.0 ], [ 0.9, 1.0 ] ],
          [ [ 0.0, 0.6 ], [ -0.75, 0.9 ], [ 0.75, 0.9 ] ],
          [ [ -0.5, 1.0 ], [ 0.5, 1.0 ] ],
          [ [ -0.533, 0.533 ], [ -0.0, 0.533 ], [ 0.533, 0.533 ] ],
          [ [ -0.3, 0.6 ], [ 0.3, 0.6 ] ],
    	  [[0,1.0]]
        ]
      }
    }"""
    vfs['cookiecutters.json']=cookiecutters_json
    return (cookiecutters_json,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### cookiecutters.js""")
    return


@app.cell
def _(cookiecutters_json, vfs):
    cookiecutters_js=r"export const cookiecutters="+cookiecutters_json+";"
    vfs["cookiecutters.js"]=cookiecutters_js
    return (cookiecutters_js,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##""")
    return


@app.cell
def _(vfs):
    list(vfs.keys()) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Conversion ES6 -> IIFE""")
    return


@app.cell
def _(
    ES6converter,
    cookiecutters_js,
    geometry_js,
    perf_counter,
    turtle_graphics_js,
    vfs,
    webgl_torus_html,
    webgl_torus_js,
):
    (webgl_torus_html,webgl_torus_js,geometry_js,turtle_graphics_js,cookiecutters_js) #dependencies
    print(f'{(len1:=len(vfs["webgl-torus.js"]))=}\n{(len2:=len(vfs["m4_cMaj.js"])) = }\n{len1+len2 = }')
    print()
    t1=perf_counter()
    iife_script=ES6converter.convertES6toIIFE('import "./webgl-torus.js";',minify=True,open=vfs.open)
    print("HTML processing completed with ES6 modules converted to IIFE functions.")
    t2=perf_counter()
    print()
    print(f'{t2-t1=}s')
    print()
    print(f'{len(iife_script)=}')
    print()
    print('IIFE JavaScript:\n',iife_script[:500], ' ...') 
    #converting the html file with ES6 module references into a single html file with minified iife functions.   
    return (iife_script,)


@app.cell
def _(mo):
    write_index_file_btn=mo.ui.run_button(label='Write "index.html" to disk.')
    return


@app.cell
def _():
    return


@app.cell
def _(iife_script, vfs, webgl_torus_html):
    #mo.stop(not write_index_file_btn.value,write_index_file_btn)
    #copy index.html from virtual file system to disk
    print('"webgl-torus.html" with ES6 module references  -->  stand alone minified "index.html" ')
    #ES6converter.process_html('webgl-torus.html',minify=False,output_file='index.html',open=vfs.open)
    #index_html=vfs['index.html'] 
    _ss=webgl_torus_html.find('<script')
    _se=webgl_torus_html.find('</script>')
    index_html=webgl_torus_html[:_ss]+'<script>'+iife_script+webgl_torus_html[_se:]
    vfs['index.html']=index_html
    print(index_html[:500])
    with open('index.html','w') as f: f.write(index_html)
    return (index_html,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Interactive 3D Viewer
    An ipywidget.Output widget is used in combination with IPython.Display.JavaScript to show the interactive webgl viewer.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### class TorusWidget""")
    return


@app.cell
def _(anywidget, iife_script, traitlets):
    class TorusWidget(anywidget.AnyWidget):
        _esm = r"""
        function render({ model, el:element }) {
            // Create debug div as an overlay
            const logdiv = document.createElement('div');
            logdiv.style.position = 'absolute';
            logdiv.style.top = '10px';
            logdiv.style.right = '10px';
            logdiv.style.maxWidth = '200px';
            logdiv.style.maxHeight = '100px';
            logdiv.style.overflow = 'auto';
            logdiv.style.background = 'rgba(255, 255, 255, 0.8)';
            logdiv.style.border = '1px solid red';
            logdiv.style.padding = '5px';
            logdiv.style.fontSize = '12px';
            logdiv.style.zIndex = '20';
            logdiv.innerText = 'Log: Init...';
            element.style.position = 'relative'; // Ensure el is a positioning context
            element.appendChild(logdiv);

            // Wait for DOM to be ready
            function initializeWidget() {
                """ + iife_script + r"""
                logdiv.innerText += '\nLog: IIFE loaded';
                const {state, drawScene, setAnimationState, canvas, animBtn, style} = modules['webgl-torus.js'];

                // Validate canvas visibility
                if (canvas.offsetWidth === 0 || canvas.offsetHeight === 0) {
                    logdiv.innerText += '\nLog: Canvas invisible!';
                    console.error('Canvas has zero size: offsetWidth=' + canvas.offsetWidth + ', offsetHeight=' + canvas.offsetHeight);
                }

                // Validate WebGL context
                const gl = canvas.getContext('webgl');
                if (!gl) {
                    logdiv.innerText += '\nLog: WebGL failed!';
                    console.error('WebGL context initialization failed');
                    return;
                }

                // Set canvas size explicitly
                canvas.width = 400;
                canvas.height = 400;
                canvas.style.width = '400px';
                canvas.style.height = '400px';
                logdiv.innerText += '\nLog: Canvas set';

                // Initialize traitlets
                console.log('Initial animate:', model.get('animate'));
                console.log('Initial hide_button:', model.get('hide_button'));
                style.hideButton = model.get('hide_button');
                setAnimationState(model.get('animate'));
    //            animBtn.style.display = style.hideButton ? 'none' : 'block';
                logdiv.innerText += '\nLog: Traitlets set';

                // Traitlet change handlers
                model.on('change:animate', () => {
                    console.log('animate changed:', model.get('animate'));
                    setAnimationState(model.get('animate'));
                    logdiv.innerText += '\nLog: animate=' + model.get('animate');
                });
                model.on('change:hide_button', () => {
                    console.log('hide_button changed:', model.get('hide_button'));
                    style.hideButton = model.get('hide_button');
                    animBtn.style.display = style.hideButton ? 'none' : 'block';
                    logdiv.innerText += '\nLog: hide_button=' + model.get('hide_button');
                });

                // Event handlers
                canvas.addEventListener('touchstart', (e) => {
                    e.preventDefault();
                    console.log('Touchstart: setting animate to false');
                    model.set('animate', false);
                    model.save_changes();
                });
                canvas.addEventListener('mousedown', (e) => {
                    console.log('Mousedown: setting animate to false');
                    model.set('animate', false);
                    model.save_changes();
                });
                animBtn.addEventListener('click', (e) => {
                    console.log('Button click: setting animate to true');
                    model.set('animate', true);
                    model.save_changes();
                });

                // Force initial render
    //            drawScene(); //causes the animation rate to double
    //            logdiv.innerText += '\nLog: Rendered';
            }

            // Delay initialization until DOM is ready
            if (document.readyState === 'complete' || document.readyState === 'interactive') {
                initializeWidget();
            } else {
                document.addEventListener('DOMContentLoaded', initializeWidget);
            }
        }
        export default {render};
        """
        animate = traitlets.Bool(default_value=True).tag(sync=True)
        hide_button = traitlets.Bool(default_value=False).tag(sync=True)

    torus_widget = TorusWidget()
    torus_widget2 = TorusWidget()
    return torus_widget, torus_widget2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Test TorusWidget""")
    return


@app.cell
def _(mo, torus_widget, torus_widget2):
    mo.hstack([torus_widget, torus_widget2])
    return


@app.cell
def _(mo, torus_widget):
    mo.ui.anywidget(torus_widget)
    return


@app.cell
def _(torus_widget):
    torus_widget.hide_button=False
    torus_widget.hide_button
    return


@app.cell
def _(torus_widget, torus_widget2):
    torus_widget.animate,torus_widget2.animate
    return


@app.cell
def _(btn):
    btn
    return


@app.cell
def _(mo, torus_widget):
    def toggle_animation(_):
        torus_widget.animate = not torus_widget.animate

    btn = mo.ui.button(label="Toggle Animation", on_click=toggle_animation)
    return (btn,)


@app.cell
def _(index_html, mo):
    mo.iframe(index_html)
    return


@app.cell
def _(mo):
    export_vfs_button=mo.ui.run_button(label='update vfs export')
    return (export_vfs_button,)


@app.cell
def _(base64, export_vfs_button, index_html, mo, vfs):
    mo.stop(not export_vfs_button.value, export_vfs_button)
    (index_html)
    def download_data(filename,data):
            b64_data = base64.b64encode(data).decode()
            download_link = f'<a download="{filename}" href="data:application/zip;base64,{b64_data}" style="display:block;">Click to download {filename}</a>' 
            return mo.iframe(download_link,height="30")
    download_data('anywidget_torus.zip',vfs.archive(filename_prefix='anywidget_torus'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Python Tests""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Segments2Complex""")
    return


@app.cell
def _(exp, inf, pi):
    def Segments2Complex(Segs,p0=0.+0.j,scale=1.0,a0=0+1j,tol=0.05,offs=0,loops=1,return_start=False):
      """
      The parameter "tol defines the resolution. It is the maximum allowable
      difference between circular arc segment, and the secant between the
      calculated points on the arc. Smaller values for tol will result in
      more points per segment.
      """
      a=a0
      p=p0*scale
      p-=1j*a*offs
      L=0
      if return_start:
          yield p,a,L,-1 #assuming closed loop: start-point = end-point
      loopcount=0
      while (loops==None) or (loops==inf) or (loopcount<loops):
          loopcount+=1
          for X,(l,da,*_) in enumerate(Segs):
            l=l*scale
            if da!=0:
              r=l/da
              r+=offs
              if r!=0:
                l=r*da
                dl=2*abs(2*r*tol)**0.5
                n=max(int(abs(6*(da/(2*pi)))),int(l//dl)+1)
              else:
                n=1
              dda=exp(1j*da/n)
              dda2=dda**0.5
              v=(2*r*dda2.imag)*dda2*a
            else:
              n=1
              dda=1
              v=l*a
            for i in range(n):
              L+=l/n
              p+=v
              v*=dda
              a*=dda
              yield p,a,L,X
    return (Segments2Complex,)


@app.cell
def _(mo):
    run_python_tests_btn=mo.ui.run_button(label='run Python tests')
    return (run_python_tests_btn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Run Python Tests""")
    return


@app.cell
def _(Segments2Complex, cookiecutters_json, mo, pi, run_python_tests_btn):
    mo.stop(not run_python_tests_btn.value, run_python_tests_btn)

    def _():
        from matplotlib import pyplot as plt
        import cmath
        import json

        def getoutline(name):
            tp = json.loads(cookiecutters_json)["outlines"][name]
            p0 = complex(*tp.get("startPoint", [0, 0]))
            a0 = cmath.rect(1.0, tp.get("startAngle", 0.0) * pi / 180)
            segs = [[l, a * pi / 180] for l, a, *_ in tp["turtlePath"]]
            return p0, a0, segs

        fig, [ax1, ax2] = plt.subplots(1, 2, width_ratios=(3.5, 1), figsize=(8, 6))
        tol = 0.05  
        p0, a0, segs = getoutline("Duck")
        p = list(zip(*[(p.real, p.imag) for p, *_ in Segments2Complex( segs, p0=p0, a0=a0, scale=2.4, tol=tol, return_start=True)]))
        Duck_points = len(p[0])-1
        ax1.plot(*p)

        ax1.set_aspect("equal")
        p0, a0, segs = getoutline("Blade")
        p = [(p.real, p.imag, h.real, h.imag) for p, h, *_ in Segments2Complex(segs, p0=p0, a0=a0, tol=tol, return_start=True)]
        for i, p_ in enumerate(p):
            print(f"{i:3}, {p_[0]:7.3f}, {p_[1]:7.3f}, {p_[2]:7.3f}, {p_[3]:7.3f}")
        Blade_points = len(p)-1
        ax2.plot(*(list(zip(*p))[:2]))
        offs=0.5
        ax2.plot(*(list(zip(*[[x+offs*sa,y-offs*ca] for x,y,ca,sa in p]))))
        ax2.set_aspect('equal')
        print(f"{Duck_points=}, {Blade_points=},{Duck_points*Blade_points=}")
        return fig
    _()
    return


@app.cell
def _(mo):
    mo.md(r"""## Trackpad""")
    return


@app.cell
def _(mo):
    mo.md(r"""### trackpad.js""")
    return


@app.cell
def _(vfs):
    trackpad_js = r"""
    import { TouchHandler } from './interaction.js';
    export function initTrackpad(canvas, logdiv, model) {
      const ctx = canvas.getContext('2d');
      let sx, sy, ex, ey, len = 0, ang = 0; // Segment start/end, length, angle
      let segs = []; // Segment array
      let editIdx = -1; // Editing index (-1: end, >=0: segment)

      // Resize canvas to fit container
      const resize = () => {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        draw();
      };
      window.addEventListener('resize', resize);
      resize();

      const draw = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Draw segments
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        let x = canvas.width / 2, y = canvas.height / 2, h = 0;
        segs.forEach(s => {
          const r = s.angle ? Math.abs(s.length / (s.angle * Math.PI / 180)) : 0;
          const steps = s.angle ? Math.max(10, Math.floor(Math.abs(s.length) / 5)) : 1;
          const sl = s.length / steps;
          const sa = (s.angle * Math.PI / 180) / steps;
          ctx.beginPath();
          ctx.moveTo(x, y);
          for (let i = 0; i < steps; i++) {
            x += sl * Math.cos(h);
            y += sl * Math.sin(h);
            h += sa;
            ctx.lineTo(x, y);
          }
          ctx.stroke();
        });
        // Draw preview
        if (sx !== undefined && ex !== undefined) {
          ctx.strokeStyle = 'gray';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          const r = ang ? Math.abs(len / ang) : 0;
          const steps = ang ? Math.max(10, Math.floor(Math.abs(len) / 5)) : 1;
          const sl = len / steps;
          const sa = ang / steps;
          let px = sx, py = sy, ph = editIdx >= 0 ? segs[editIdx].h : h;
          for (let i = 0; i < steps; i++) {
            px += sl * Math.cos(ph);
            py += sl * Math.sin(ph);
            ph += sa;
            ctx.lineTo(px, py);
          }
          ctx.stroke();
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 2;
        }
        // Draw cursors
        ctx.beginPath();
        ctx.arc(sx || canvas.width / 2, sy || canvas.height / 2, 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();
        ctx.stroke();
        if (ex !== undefined) {
          ctx.beginPath();
          ctx.arc(ex, ey, 5, 0, 2 * Math.PI);
          ctx.fillStyle = 'green';
          ctx.fill();
          ctx.stroke();
        }
      };

      new TouchHandler(canvas, {
        start: (x, y) => {
          sx = x;
          sy = y;
          ex = x;
          ey = y;
          len = 0;
          ang = 0;
          if (editIdx === -1) {
            segs.push({ length: 0, angle: 0, h: segs.length ? segs[segs.length - 1].h : 0 });
            editIdx = segs.length - 1;
          }
          model.set('animate', false);
          model.save_changes();
          logdiv.innerText = `Trackpad Log:\nStart: (${x.toFixed(1)}, ${y.toFixed(1)})`;
          draw();
        },
        move: (dx, dy, x, y) => {
          ex = x;
          ey = y;
          const sdx = x - sx;
          const sdy = y - sy;
          len = Math.hypot(sdx, sdy);
          const sa = Math.atan2(sdy, sdx);
          const h = editIdx >= 0 ? segs[editIdx].h : (segs.length ? segs[segs.length - 1].h : 0);
          ang = (sa - h) * 2;
          ang = ((ang + 2 * Math.PI + 4 * Math.PI) % (4 * Math.PI)) - 2 * Math.PI;
          if (editIdx >= 0) {
            segs[editIdx] = { length: len, angle: ang * 180 / Math.PI, h };
          }
          logdiv.innerText = `Trackpad Log:\nStart: (${sx.toFixed(1)}, ${sy.toFixed(1)})\nMove: dx=${dx.toFixed(1)}, dy=${dy.toFixed(1)} (3D rot), len=${len.toFixed(1)}, ang=${(ang * 180 / Math.PI).toFixed(1)}° (2D)`;
          draw();
        },
        end: () => {
          if (editIdx === -1) editIdx = segs.length - 1;
          sx = undefined;
          ex = undefined;
          model.set('animate', false);
          model.save_changes();
          logdiv.innerText = `Trackpad Log:\nEnd: Segment: ${len.toFixed(1)}px, ${(ang * 180 / Math.PI).toFixed(1)}°`;
          draw();
        },
        zoom: (ds) => {
          logdiv.innerText = `Trackpad Log:\nZoom: ${ds.toFixed(3)}`;
        },
        pan: (dx, dy) => {
          logdiv.innerText = `Trackpad Log:\nPan: (${dx.toFixed(1)}, ${dy.toFixed(1)})`;
        }
      });
      draw();
    }
    """
    vfs["trackpad.js"]=trackpad_js
    return (trackpad_js,)


@app.cell
def _(mo):
    mo.md(r"""### class TrackpadWidget""")
    return


@app.cell
def _(ES6converter, trackpad_js, vfs):
    (trackpad_js)
    trackpad_iife_js=ES6converter.convertES6toIIFE(r'import "./trackpad.js";', minify=False,open=vfs.open)
    return (trackpad_iife_js,)


@app.cell
def _(anywidget, trackpad_iife_js, traitlets):
    class TrackpadWidget(anywidget.AnyWidget):
        _esm = trackpad_iife_js+r"""
        function render({ model, el }) {
          const log = document.createElement('div');
          log.style.position = 'absolute';
          log.style.top = '10px';
          log.style.right = '10px';
          log.style.maxWidth = '200px';
          log.style.maxHeight = '100px';
          log.style.overflow = 'auto';
          log.style.background = 'rgba(255, 255, 255, 0.8)';
          log.style.border = '1px solid red';
          log.style.padding = '5px';
          log.style.fontSize = '12px';
          log.style.zIndex = '20';
          log.innerText = 'Trackpad Log:';
          const c = document.createElement('canvas');
          c.style.width = '400px';
          c.style.height = '400px';
          el.style.position = 'relative';
          el.appendChild(c);
          el.appendChild(log);
          // Ensure IIFE modules are loaded
          if (!window.modules || !window.modules['trackpad.js']) {
            log.innerText += '\nError: trackpad.js not loaded';
            return;
          }
          const { initTrackpad } = window.modules['trackpad.js'];
          initTrackpad(c, log, model);
        }
        export default {render};
        """
        animate = traitlets.Bool(default_value=True).tag(sync=True)
    trackpad_widget = TrackpadWidget()
    return


@app.cell
def _(anywidget, traitlets, vfs):
    def _():
        iife_prefix = r"""
        (function(global) {
          class TouchHandler {
            constructor(c, cb) {
              if (!c || !(c instanceof HTMLCanvasElement)) {
                console.error('TouchHandler: Invalid canvas', c);
                throw new Error('Invalid canvas');
              }
              this.c = c;
              this.cb = cb || {};
              this.d = false;
              this.x = 0;
              this.y = 0;
              this.init();
            }
            init() {
              console.log('TouchHandler: Initializing on canvas', this.c);
              const evts = [
                ['touchstart', e => this.hS(e)],
                ['touchmove', e => this.hM(e)],
                ['touchend', e => this.hE(e)],
                ['mousedown', e => this.hMD(e)],
                ['mousemove', e => this.hMM(e)],
                ['mouseup', e => this.hMU(e)]
              ];
              evts.forEach(([t, f]) => {
                this.c.addEventListener(t, f.bind(this));
                console.log('Bound event', t);
              });
              document.addEventListener('touchmove', e => {
                if (e.target === this.c) e.preventDefault();
              }, { passive: false });
            }
            pos(e) {
              const r = this.c.getBoundingClientRect();
              const t = e.touches?.[0] || e.changedTouches?.[0] || e;
              const x = t.clientX - r.left;
              const y = t.clientY - r.top;
              console.log('pos', { clientX: t.clientX, left: r.left, x, clientY: t.clientY, top: r.top, y });
              return { x, y };
            }
            hS(e) {
              e.preventDefault();
              this.d = true;
              const { x, y } = this.pos(e);
              this.x = x;
              this.y = y;
              console.log('touchstart', { x, y });
              if (this.cb.start) this.cb.start(x, y);
            }
            hM(e) {
              e.preventDefault();
              if (!this.d) return;
              const { x, y } = this.pos(e);
              const dx = x - this.x;
              const dy = y - this.y;
              console.log('touchmove', { x, y, dx, dy });
              if (this.cb.move) this.cb.move(dx, dy, x, y);
              this.x = x;
              this.y = y;
            }
            hE(e) {
              e.preventDefault();
              this.d = false;
              console.log('touchend');
              if (this.cb.end) this.cb.end();
            }
            hMD(e) {
              this.d = true;
              const { x, y } = this.pos(e);
              this.x = x;
              this.y = y;
              console.log('mousedown', { x, y });
              if (this.cb.start) this.cb.start(x, y);
            }
            hMM(e) {
              if (!this.d) return;
              const { x, y } = this.pos(e);
              const dx = x - this.x;
              const dy = y - this.y;
              console.log('mousemove', { x, y, dx, dy });
              if (this.cb.move) this.cb.move(dx, dy, x, y);
              this.x = x;
              this.y = y;
            }
            hMU(e) {
              this.d = false;
              console.log('mouseup');
              if (this.cb.end) this.cb.end();
            }
          }
          if (!('modules' in global)) {
            global['modules'] = {};
          }
          global.modules['interaction.js'] = { TouchHandler };
        })(window);
        (function(global) {
          let { TouchHandler } = global.modules['interaction.js'];
          function initTrackpad(canvas, logdiv, model) {
            console.log('Trackpad: Starting initTrackpad');
            const ctx = canvas.getContext('2d');
            if (!ctx) {
              console.error('Trackpad: Failed to get 2D context');
              logdiv.innerText += '\nError: Failed to get 2D context';
              return;
            }
            logdiv.innerText += '\nTrackpad: Context initialized';
            canvas.width = 400;
            canvas.height = 400;
            console.log('Trackpad: Set canvas size', { width: canvas.width, height: canvas.height });
            let x, y;
            const draw = () => {
              try {
                console.log('Trackpad: Drawing');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                logdiv.innerText += '\nTrackpad: Cleared canvas';
                if (x !== undefined && y !== undefined) {
                  ctx.beginPath();
                  ctx.arc(x, y, 5, 0, 2 * Math.PI);
                  ctx.fillStyle = 'red';
                  ctx.fill();
                  ctx.strokeStyle = 'black';
                  ctx.stroke();
                  logdiv.innerText += '\nTrackpad: Drew red dot';
                }
              } catch (err) {
                console.error('Trackpad: Draw error', err);
                logdiv.innerText += `\nTrackpad: Draw error: ${err.message}`;
              }
            };
            new TouchHandler(canvas, {
              start: (px, py) => {
                console.log('Trackpad: start callback', { x: px, y: py });
                x = px;
                y = py;
                logdiv.innerText += `\nStart: (${px.toFixed(1)}, ${py.toFixed(1)})`;
                draw();
              },
              move: (dx, dy, px, py) => {
                console.log('Trackpad: move callback', { dx, dy, x: px, y: py });
                x = px;
                y = py;
                logdiv.innerText += `\nMove: (${px.toFixed(1)}, ${py.toFixed(1)})`;
                draw();
              },
              end: () => {
                console.log('Trackpad: end callback');
                x = undefined;
                y = undefined;
                logdiv.innerText += '\nEnd';
                draw();
              }
            });
            console.log('Trackpad: Initialized');
            draw();
          }
          if (!('modules' in global)) {
            global['modules'] = {};
          }
          global.modules['trackpad.js'] = { initTrackpad };
        })(window);
        """
        vfs["trackpad.js"] = r"""
        import { TouchHandler } from './interaction.js';
        export function initTrackpad(canvas, logdiv, model) {
          console.log('Trackpad: Starting initTrackpad');
          const ctx = canvas.getContext('2d');
          if (!ctx) {
            console.error('Trackpad: Failed to get 2D context');
            logdiv.innerText += '\nError: Failed to get 2D context';
            return;
          }
          logdiv.innerText += '\nTrackpad: Context initialized';
          canvas.width = 400;
          canvas.height = 400;
          console.log('Trackpad: Set canvas size', { width: canvas.width, height: canvas.height });
          let x, y;

          const draw = () => {
            try {
              console.log('Trackpad: Drawing');
              ctx.clearRect(0, 0, canvas.width, canvas.height);
              logdiv.innerText += '\nTrackpad: Cleared canvas';
              if (x !== undefined && y !== undefined) {
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI);
                ctx.fillStyle = 'red';
                ctx.fill();
                ctx.strokeStyle = 'black';
                ctx.stroke();
                logdiv.innerText += '\nTrackpad: Drew red dot';
              }
            } catch (err) {
              console.error('Trackpad: Draw error', err);
              logdiv.innerText += `\nTrackpad: Draw error: ${err.message}`;
            }
          };

          new TouchHandler(canvas, {
            start: (px, py) => {
              console.log('Trackpad: start callback', { x: px, y: py });
              x = px;
              y = py;
              logdiv.innerText += `\nStart: (${px.toFixed(1)}, ${py.toFixed(1)})`;
              draw();
            },
            move: (dx, dy, px, py) => {
              console.log('Trackpad: move callback', { dx, dy, x: px, y: py });
              x = px;
              y = py;
              logdiv.innerText += `\nMove: (${px.toFixed(1)}, ${py.toFixed(1)})`;
              draw();
            },
            end: () => {
              console.log('Trackpad: end callback');
              x = undefined;
              y = undefined;
              logdiv.innerText += '\nEnd';
              draw();
            }
          });
          console.log('Trackpad: Initialized');
          draw();
        }
        """
        class TrackpadWidget(anywidget.AnyWidget):
            _esm = iife_prefix + r"""
            function render({ model, el }) {
              console.log('TrackpadWidget: Starting render');
              const log = document.createElement('div');
              log.style.position = 'absolute';
              log.style.top = '10px';
              log.style.right = '10px';
              log.style.maxWidth = '200px';
              log.style.maxHeight = '100px';
              log.style.overflow = 'auto';
              log.style.background = 'rgba(255, 255, 255, 0.8)';
              log.style.border = '1px solid red';
              log.style.padding = '5px';
              log.style.fontSize = '12px';
              log.style.zIndex = '20';
              log.innerText = 'Trackpad Log:';
              const c = document.createElement('canvas');
              c.style.width = '400px';
              c.style.height = '400px';
              el.style.position = 'relative';
              el.appendChild(c);
              el.appendChild(log);
              log.innerText += '\nTrackpadWidget: Canvas created';
              if (!window.modules || !window.modules['trackpad.js']) {
                log.innerText += '\nError: trackpad.js not loaded';
                console.error('TrackpadWidget: Modules not loaded', window.modules);
                return;
              }
              const { initTrackpad } = window.modules['trackpad.js'];
              console.log('TrackpadWidget: Calling initTrackpad');
              initTrackpad(c, log, model);
              log.innerText += '\nTrackpadWidget: initTrackpad called';
            }
            export default {render};
            """
            animate = traitlets.Bool(default_value=True).tag(sync=True)
        trackpad_widget = TrackpadWidget()
        return trackpad_widget


    _()
    return


@app.cell
def _(vfs):
    print(list(vfs.keys()))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
