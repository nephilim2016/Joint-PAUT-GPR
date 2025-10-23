#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 07:22:05 2025

@author: nephilim
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import numpy as np


class ImageSelector:
    def __init__(self, ax, data):
        self.ax = ax
        self.data = data
        self.filled_data = None  # This will store the filled data
        self.cax = ax.imshow(data, cmap='viridis')
        self.polygon_selector = PolygonSelector(ax, self.onselect, useblit=True)
        ax.set_title('Select a closed area and press Enter to confirm')
        
        # Connect the key event to finish selection
        self.cid = ax.figure.canvas.mpl_connect('key_press_event', self.on_key)
    
    def onselect(self, verts):
        """Handle polygon selection and create a mask."""
        path = Path(verts)
        x, y = np.meshgrid(np.arange(self.data.shape[1]), np.arange(self.data.shape[0]))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        mask = path.contains_points(points)
        mask = mask.reshape(self.data.shape)
        
        # Fill the selected area with a value (e.g., 1)
        self.filled_data = self.data.copy()
        self.filled_data[mask] = 3  # Set selected area to 1
        self.cax.set_data(self.filled_data)
        plt.draw()
    
    def on_key(self, event):
        """Handle key press events (e.g., 'enter' to finish selection)."""
        if event.key == 'enter':
            print("Selection finished. You can now use the filled data.")
            self.disconnect()
    
    def disconnect(self):
        """Disconnect all events."""
        self.polygon_selector.disconnect_events()
        self.ax.figure.canvas.mpl_disconnect(self.cid)
        self.ax.set_title('Selection completed')
        self.ax.figure.canvas.draw()

if __name__=='__main__':
    # Sample color image data
    # data = np.load('RTM.npy')[10:-10,10:-10]/np.load('Source1.npy')[10:-10,10:-10]
    # data=np.load('correlation_rtm_result.npy')
    # data=data[10:-10,10:-10]
    data=filled_data
    # Create the plot
    fig, ax = plt.subplots()
    selector = ImageSelector(ax, data)
    
    # Show the plot in non-blocking mode
    plt.show(block=True)
    
    # After the plot window is closed, access the filled data
    filled_data = selector.filled_data
    if filled_data is not None:
        print("Filled data is available.")
        # Display the filled data
        plt.figure()
        plt.imshow(filled_data, cmap='viridis')
        plt.title('Filled Data')
        plt.show()
    else:
        print("No selection was made or filled data is not available.")

