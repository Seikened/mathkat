#!/bin/bash

# Define la versión automáticamente (puedes ajustarla o obtenerla de otra fuente)
VERSION="v1.6.0"

echo "Creando una nueva etiqueta: $VERSION"
git tag "$VERSION"

echo "Subiendo la etiqueta al repositorio remoto"
git push origin "$VERSION"