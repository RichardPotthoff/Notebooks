#ipynb_marimo_wasm.py
#This python script uses marimo to first convert an IPython notebook
#inot a marimo notebook, and then convert the marimo notebook into a
#static wasm html that can be viewed interactively in a web browser.
import marimo as mo
from marimo.__main__ import main as marimo_main
import sys,os

if len(sys.argv)>1:
  ipynbPath=sys.argv[1]
else:
  ipynbPath='Counter.ipynb'
  
ipynbName,ipynbExt=os.path.splitext(ipynbPath)

script=[f'marimo -l DEBUG convert {ipynbName+(ipynbExt or ".ipynb")} -o {ipynbName+".py"}',
        f'marimo -l DEBUG export html-wasm {ipynbName+".py"} -o {ipynbName}']
        
for cmdln in script:
  try: 
    sys.argv=cmdln.split()
    print(f"running: {' '.join(sys.argv)}\n ")
    marimo_main()
  except BaseException as e:
    print(f"completed with code: {e}\n")


import webbrowser
from http.server import HTTPServer,SimpleHTTPRequestHandler
from threading import Thread

curdir=os.getcwd()
os.chdir(ipynbName)

Server=HTTPServer(('localhost',8088),SimpleHTTPRequestHandler)
ServerThread=Thread(target=lambda:Server.serve_forever())
ServerThread.daemon=True
ServerThread.start()

webbrowser.open('http://localhost:8088/index.html')

try:
  while True:
    pass
except KeyboardInterrupt:
  Server.shutdown()
  ServerThread.join()
  
os.chdir(curdir)
