"""Make `import omni.physxdemos` work again on Isaac Sim 5.1.0.

Env_Config/Garment/Particle_Garment.py and Env_Config/Human/Human.py both do
`import omni.physxdemos as demo` at module scope. On Isaac Sim 4.5.0 that
module happened to be importable out of the box; on 5.1.0 the extension that
provides it (`omni.physx.demos`) is present in the registry but not enabled by
the default app config, so the bare import raises ModuleNotFoundError before
either file's class body is even reached.

Neither file ever actually USES the module -- `demo.` appears nowhere in the
repo -- so this is a dead import, but it still has to resolve.

Rather than editing the user's source (the point of this container is to leave
the host repo untouched), a meta-path finder is installed here that enables the
Kit extension the first time anything asks for `omni.physxdemos`, then hands the
import back to the normal machinery. It has to be lazy like this because the
extension manager does not exist until SimulationApp has been constructed,
which happens long after interpreter startup.
"""
import importlib.machinery
import importlib.util
import sys


class _PhysxDemosFinder:
    """Enable omni.physx.demos on first request for omni.physxdemos."""

    TARGET = "omni.physxdemos"
    EXTENSION = "omni.physx.demos"

    def __init__(self):
        self._tried = False

    def find_spec(self, fullname, path=None, target=None):
        # Only ever act once: after enabling, the real PathFinder ahead of us in
        # sys.meta_path resolves the module, and re-entering here (find_spec
        # below walks sys.meta_path again) would recurse.
        if fullname != self.TARGET or self._tried:
            return None
        self._tried = True

        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            if app is None:
                return None
            mgr = app.get_extension_manager()
            mgr.set_extension_enabled_immediate(self.EXTENSION, True)
        except Exception:  # noqa: BLE001
            # No Kit app yet, or the extension is gone in this version. Fall
            # through to a stub rather than taking the whole env down over an
            # import the repo never uses.
            return self._stub_spec(fullname)

        # Enabling put the extension's own directory on sys.path; `omni` is a
        # namespace package so its __path__ picks that up automatically.
        import omni

        spec = importlib.machinery.PathFinder.find_spec(fullname, list(omni.__path__))
        return spec if spec is not None else self._stub_spec(fullname)

    @staticmethod
    def _stub_spec(fullname):
        loader = importlib.machinery.SourceFileLoader(fullname, __file__)
        spec = importlib.util.spec_from_loader(fullname, loader=None)
        spec.loader = _StubLoader()
        return spec


class _StubLoader:
    def create_module(self, spec):
        import types

        mod = types.ModuleType(spec.name)
        mod.__doc__ = ("Stub standing in for omni.physxdemos, which this Isaac Sim "
                       "build does not provide. The importing code never calls into it.")
        return mod

    def exec_module(self, module):
        return None


if not any(isinstance(f, _PhysxDemosFinder) for f in sys.meta_path):
    sys.meta_path.append(_PhysxDemosFinder())
