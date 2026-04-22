# VoxelMorph Shape Morphing

A from-scratch reimplementation of the core VoxelMorph registration method,
adapted for 3D shape deformation and animation in Blender.

## What this does

Instead of medical images, this project applies the VoxelMorph approach to
simple voxelized 3D shapes. A lightweight U-Net learns the displacement field
between two shapes, and the resulting field is exported to Blender to generate
a smooth morphing animation.

![L-Shape displacement field for a Cube](https://s7.ezgif.com/tmp/ezgif-7c26ac259c1afeed.gif)
![L-Shape displacement field for a Sphere](https://s7.ezgif.com/tmp/ezgif-78d3a5f49f4eb2a0.gif)
![Shrink and Shift displacement field for a Cube](https://s7.ezgif.com/tmp/ezgif-7b0a96d7150e987a.gif)
![Shrink and Shift displacement field for a Sphere](https://s7.ezgif.com/tmp/ezgif-7283e9250adc1474.gif)

## Pipeline

Shape A + Shape B → Mini U-Net (PyTorch) → Displacement field → Blender animation

## Project structure

├── model.py              # U-Net architecture + spatial transformer
├── train.py              # Voxel grid creation and training loop
├── export_to_blender.py  # Saves displacement field as .npy
└── blender_import.py     # Applies field to mesh as shape keys

## Slides

📊 [View full presentation](https://etusorbonneuniversitefr-my.sharepoint.com/:p:/g/personal/yasaman_tavakkoli_tabasi_etu_sorbonne-universite_fr/IQB4tqpN-iP9QIGeZaf2KlewASNLquNuoQzP40IUjN4F1WE?e=7l7bXT)

## Inspiration

Based on the VoxelMorph paper:
Balakrishnan et al., "An Unsupervised Learning Model for Deformable Medical
Image Registration", CVPR 2018.
