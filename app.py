import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.gradio_app import demo

demo.launch()