# VoxelMorph Shape Morphing

A from-scratch reimplementation of the core VoxelMorph registration method,
adapted for 3D shape deformation and animation in Blender.

## What this does

Instead of medical images, this project applies the VoxelMorph approach to
simple voxelized 3D shapes. A lightweight U-Net learns the displacement field
between two shapes, and the resulting field is exported to Blender to generate
a smooth morphing animation.

## Pipeline

Shape A + Shape B → Mini U-Net (PyTorch) → Displacement field → Blender animation

## Project structure

├── model.py              # U-Net architecture + spatial transformer
├── train.py              # Voxel grid creation and training loop
├── export_to_blender.py  # Saves displacement field as .npy
└── blender_import.py     # Applies field to mesh as shape keys

## Inspiration

Based on the VoxelMorph paper:
Balakrishnan et al., "An Unsupervised Learning Model for Deformable Medical
Image Registration", CVPR 2018.
