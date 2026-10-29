"""Découverte automatique des modules : un module = un dossier modules/<id>/ avec un fichier module.py."""
import importlib
from pathlib import Path
from .team import TEAM

MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"


def discover():
    """Retourne {id: module_python | Exception}. Un module cassé n'empêche pas les autres de s'afficher."""
    found = {}
    for t in TEAM:
        if (MODULES_DIR / t["id"] / "module.py").exists():
            try:
                mod = importlib.import_module(f"modules.{t['id']}.module")
                assert hasattr(mod, "META") and callable(getattr(mod, "render", None)), "module.py doit définir META et render()"
                found[t["id"]] = mod
            except Exception as e:  # noqa: BLE001
                found[t["id"]] = e
    return found
