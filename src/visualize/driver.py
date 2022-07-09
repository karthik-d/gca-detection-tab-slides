# Pipeline the visualization steps here, for each phase

from .explainer import gradcam 

def explain():
    gradcam.save_visualization()

def visualize():
    explain()