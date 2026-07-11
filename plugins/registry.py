"""
Central plugin registry with auto-discovery.

On startup, discover() scans plugins/<category>/ directories, imports
each .py file, calls its create(settings) factory, and registers the
resulting Plugin instance.

For platform plugins, register_platforms() creates the adapter and
registers it with PlatformManager — replacing the hardcoded manual
registration that was previously in main.py.

Adding a new plugin requires ZERO changes to:
    • face_processor.py (the AI inference pipeline)
    • voice_processor.py (the voice processing chain)
    • audio_pipeline.py / frame_queue.py (the threading pipeline)
    • main.py (only the lifespan calls discover/register, never per-plugin)
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

from utils.logger import setup_logger
from plugins.base import Plugin, PlatformPlugin

logger = setup_logger("plugin_registry")

_CATEGORIES = ("platforms", "ai_models", "voice_effects", "video_effects")


class PluginRegistry:
    """Central registry for all plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._by_category: dict[str, list[str]] = {}

    # ── Registration ───────────────────────────────────────

    def register(self, plugin: Plugin) -> bool:
        """Initialize and register a plugin. Returns True on success."""
        if not plugin.name:
            logger.error("Plugin missing 'name' — skipping")
            return False

        if plugin.name in self._plugins:
            logger.warning("Plugin '%s' already registered — skipping", plugin.name)
            return False

        try:
            ok = plugin.initialize()
            if not ok:
                logger.error("Plugin '%s' initialize() returned False", plugin.name)
                return False
        except Exception as exc:
            logger.error("Plugin '%s' initialize() error: %s", plugin.name, exc)
            return False

        self._plugins[plugin.name] = plugin
        self._by_category.setdefault(plugin.category, []).append(plugin.name)
        logger.info(
            "Registered plugin: %s [%s] v%s — %s",
            plugin.name,
            plugin.category,
            plugin.version,
            plugin.display_name,
        )
        return True

    def unregister(self, name: str) -> bool:
        """Shutdown and remove a plugin."""
        plugin = self._plugins.pop(name, None)
        if plugin is None:
            return False
        try:
            plugin.shutdown()
        except Exception as exc:
            logger.error("Shutdown error (%s): %s", name, exc)
        cat = self._by_category.get(plugin.category, [])
        if name in cat:
            cat.remove(name)
        return True

    # ── Lookup ─────────────────────────────────────────────

    def get(self, name: str) -> Optional[Plugin]:
        return self._plugins.get(name)

    def get_by_category(self, category: str) -> list[Plugin]:
        names = self._by_category.get(category, [])
        return [self._plugins[n] for n in names]

    def list_all(self) -> list[dict]:
        return [p.to_dict() for p in self._plugins.values()]

    def list_category(self, category: str) -> list[dict]:
        return [p.to_dict() for p in self.get_by_category(category)]

    @property
    def count(self) -> int:
        return len(self._plugins)

    # ── Auto-discovery ─────────────────────────────────────

    def discover(self, settings=None) -> int:
        """Scan plugins/ subdirectories and auto-register all plugins.

        Each plugin file must export a create(settings=None) -> Plugin
        factory function.

        Returns the number of plugins successfully registered.
        """
        plugins_dir = Path(__file__).parent
        registered = 0

        for category_dirname in _CATEGORIES:
            category_dir = plugins_dir / category_dirname
            if not category_dir.is_dir():
                continue

            for plugin_file in sorted(category_dir.glob("*.py")):
                if plugin_file.name.startswith("_"):
                    continue

                module_path = f"plugins.{category_dirname}.{plugin_file.stem}"
                try:
                    module = importlib.import_module(module_path)
                except Exception as exc:
                    logger.error("Failed to import %s: %s", module_path, exc)
                    continue

                factory = getattr(module, "create", None)
                if factory is None:
                    logger.warning("No create() factory in %s — skipping", module_path)
                    continue

                try:
                    plugin = factory(settings)
                except Exception as exc:
                    logger.error("create() failed in %s: %s", module_path, exc)
                    continue

                if not isinstance(plugin, Plugin):
                    logger.error(
                        "%s create() returned non-Plugin: %s", module_path, type(plugin)
                    )
                    continue

                if self.register(plugin):
                    registered += 1

        logger.info("Plugin discovery complete — %d plugins registered", registered)
        return registered

    # ── Platform registration ──────────────────────────────

    def register_platforms(self, platform_manager) -> int:
        """Register all platform plugins' adapters with PlatformManager.

        This replaces the hardcoded platform_manager.register() calls
        that were previously in main.py lifespan.
        """
        count = 0
        for plugin in self.get_by_category("platform"):
            adapter = plugin.get_adapter()
            if adapter is None:
                continue
            platform_manager.register(adapter)
            count += 1
        logger.info("Registered %d platform adapters with PlatformManager", count)
        return count

    # ── Shutdown ───────────────────────────────────────────

    def shutdown_all(self) -> None:
        """Shutdown every registered plugin."""
        for plugin in list(self._plugins.values()):
            try:
                plugin.shutdown()
            except Exception as exc:
                logger.error("Shutdown error (%s): %s", plugin.name, exc)
        self._plugins.clear()
        self._by_category.clear()
        logger.info("All plugins shut down")


# Singleton
plugin_registry = PluginRegistry()