// Dynamic plugin loader for ASI engines
const fs = require('fs');
const path = require('path');

function loadPlugins(dir) {
  const plugins = {};
  if (!fs.existsSync(dir)) return plugins;
  fs.readdirSync(dir).forEach(file => {
    if (file.endsWith('.js')) {
      const pluginPath = path.join(dir, file);
      try {
        const plugin = require(pluginPath);
        plugins[file.replace('.js', '')] = plugin;
      } catch (err) {
        // Log and skip bad plugins
        // Optionally: logger.warn(`Failed to load plugin ${file}: ${err.message}`);
      }
    }
  });
  return plugins;
}

module.exports = { loadPlugins };
