# marimo.py (v0)
"""
Custom marimo loader for Pythonista on ios.

- Adds site-packages to sys.path dynamically.

- Patches sys.std* streams for marimo compatibility.

- Loads marimo explicitly.
"""
modulename = 'marimo'
modulefilepath = "/private/var/mobile/Containers/Data/Application/8C15CF35-580B-488E-9B29-D89BDB144E80/Documents/Documents/site-packages/marimo/__init__.py"
# Add site-packages to path
additional_site_packages="/private/var/mobile/Containers/Data/Application/8C15CF35-580B-488E-9B29-D89BDB144E80/Documents/Documents/site-packages"

import sys
import types
import importlib.util

print(f'Patching module "{modulename}" for iOS.',file=sys.stderr) 

if additional_site_packages and not(additional_site_packages in sys.path):
  sys.path.insert(1,additional_site_packages) #insert after the cwd

# Patch sys.std* streams
for sysstdstream, i in [(sys.stdin, 0), (sys.stdout, 1), (sys.stderr, 2)]:
  if not hasattr(sysstdstream, 'errors'):
    sysstdstream.errors = 'strict'
  if not hasattr(sysstdstream, 'fileno'):
    sysstdstream._fileno = i
    sysstdstream.fileno = types.MethodType(lambda self:self._fileno, sysstdstream)

# Load module explicitly
spec = importlib.util.spec_from_file_location(modulename, modulefilepath)
module = importlib.util.module_from_spec(spec)
sys.modules[modulename] = module
spec.loader.exec_module(module)

if __name__=='__main__':
  from marimo.__main__ import main
  main()
else:
  if ('__main__' in sys.modules) and  not(hasattr(sys.modules['__main__'],'__spec__')):
    sys.modules['__main__'].__spec__==None #necessary for running script in IPython console (for some reason)
  from marimo import *

