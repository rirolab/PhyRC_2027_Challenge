"""Preserve failing command exit status through Isaac's fast shutdown."""
import sys


def install_failure_handler():
    def failed(exc_type, value, tb):
        sys.__excepthook__(exc_type, value, tb)
        module = sys.modules.get('Env_StandAlone.Teleop_TShirt_Stretch4_Env')
        if module is not None and hasattr(module, 'simulation_app'):
            module.simulation_app.close(exit_code=1)
    sys.excepthook = failed
