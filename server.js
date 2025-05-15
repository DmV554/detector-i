const express = require('express');
const path = require('path'); // Módulo para manejar rutas de archivos
const app = express();
const port = 3000; // Puedes cambiar el puerto si lo deseas

// Middleware para añadir las cabeceras COOP y COEP
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
  next();
});

// Servir archivos estáticos desde el directorio actual (donde está index.html)
// __dirname es la ruta del directorio donde se encuentra server.js
app.use(express.static(path.join(__dirname, '')));

app.listen(port, () => {
  console.log(`Servidor escuchando en http://localhost:${port}`);
  console.log(`Sirviendo archivos desde: ${path.join(__dirname, '')}`);
  console.log('Presiona Ctrl+C para detener el servidor.');
});