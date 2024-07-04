import fs from 'fs';
import path from 'path';

// Function to delete a directory recursively
const deleteDirectory = (dirPath) => {
  if (fs.existsSync(dirPath)) {
    fs.readdirSync(dirPath).forEach((file) => {
      const currentPath = path.join(dirPath, file);
      if (fs.lstatSync(currentPath).isDirectory()) {
        deleteDirectory(currentPath);
      } else {
        fs.unlinkSync(currentPath);
      }
    });
    fs.rmdirSync(dirPath);
  }
};

// Function to copy a directory recursively
const copyDirectory = (src, dest) => {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }
  const entries = fs.readdirSync(src, { withFileTypes: true });
  for (let entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirectory(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
};

// Main script
const args = process.argv.slice(2);
switch (args[0]) {
  case 'prebuild':
    deleteDirectory(path.resolve('dist'));
    break;
  case 'copy-grammars':
    copyDirectory(
      path.resolve('src/engines/node-llama-cpp/grammars'),
      path.resolve('dist/engines/node-llama-cpp/grammars')
    );
    break;
  default:
    console.log('Unknown command');
}
