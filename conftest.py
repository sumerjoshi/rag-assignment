# pytest picks up this file at the repo root, which puts the root on sys.path.
# that lets the tests do "from src.config import ..." the same way the app does.
