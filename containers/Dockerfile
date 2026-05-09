# DOLFINx 0.9.0 with PyVista and project dependencies
# Build: docker build -t dolfinx-pyvista:0.9.0 -f Dockerfile.local .
# Run:   docker run -it --rm -v $(pwd):/work -w /work dolfinx-pyvista:0.9.0

FROM ghcr.io/fenics/dolfinx/dolfinx:v0.9.0

# Install additional Python packages
RUN pip install --no-cache-dir \
    pyvista \
    vtk \
    meshio \
    matplotlib \
    pyyaml \
    numpy \
    scipy \
    h5py \
    gmsh \
    pygmsh \
    tqdm

# Set up environment for headless rendering
ENV PYVISTA_OFF_SCREEN=true
ENV PYVISTA_USE_IPYVTK=false
ENV DISPLAY=:99

# Verify installation
RUN python3 -c "import pyvista; import dolfinx; import meshio; import matplotlib; print('All packages installed successfully')"
