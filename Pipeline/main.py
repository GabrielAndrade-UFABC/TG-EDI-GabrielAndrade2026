import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# módulo importável -> pacote do pip (o suficiente para a inferência; codebleu é opcional, só p/ métricas)
REQUIRED = {
    "torch": "torch",
    "transformers": "transformers",
    "safetensors": "safetensors",
}
YES = ("", "s", "sim", "y", "yes")
HELP_FLAGS = ("-h", "--help")


class SlottedMeta(type):
    """Recusa, já na definição, classes sem `__slots__`: instâncias sem `__dict__` e atributos fixos."""

    def __new__(mcs, name, bases, namespace):
        if "__slots__" not in namespace:
            raise TypeError(f"{name} precisa declarar __slots__.")
        return super().__new__(mcs, name, bases, namespace)


class Slotted(metaclass=SlottedMeta):
    __slots__ = ()


class DependencyCheck(Slotted):
    """Confere os pacotes da inferência e, num terminal, oferece instalá-los com o pip."""

    __slots__ = ("required", "find_spec", "install", "ask", "interactive", "executable")

    def __init__(
        self,
        required=REQUIRED,
        *,
        find_spec=importlib.util.find_spec,
        install=subprocess.call,
        ask=input,
        interactive=lambda: sys.stdin.isatty(),
        executable=sys.executable,
    ):
        self.required = required
        self.find_spec = find_spec
        self.install = install
        self.ask = ask
        self.interactive = interactive
        self.executable = executable

    def missing(self):
        return [
            pkg
            for module, pkg in self.required.items()
            if self.find_spec(module) is None
        ]

    def ensure(self):
        missing = self.missing()
        if not missing:
            return
        command = [self.executable, "-m", "pip", "install", *missing]
        print(f'Dependências ausentes: {", ".join(missing)}', file=sys.stderr)
        if self.interactive():
            if (
                self.ask(f'Instalar agora ({" ".join(command)})? [S/n] ')
                .strip()
                .lower()
                in YES
            ):
                if self.install(command) == 0 and not self.missing():
                    return
                print("A instalação falhou.", file=sys.stderr)
        raise SystemExit(
            f'Instale manualmente e rode de novo:\n  {" ".join(command)}\n'
            "Ou use: pip install -r requirements.txt"
        )


class ModelCheck(Slotted):

    __slots__ = ("root", "defaults")

    def __init__(self, root=ROOT, defaults=None):
        self.root = root
        self.defaults = defaults

    def absent(self, argv):
        if self.defaults is None:
            from pipeline import (
                DEFAULT_MODELS,  # só depois de confirmar as dependências
            )

            self.defaults = DEFAULT_MODELS
        return [
            str(path.relative_to(self.root))
            for kind, path in self.defaults.items()
            if not any(arg.startswith(f"--{kind}-model") for arg in argv)
            and not (path / "model.safetensors").is_file()
        ]

    def ensure(self, argv):
        absent = self.absent(argv)
        if absent:
            raise SystemExit(
                "Checkpoints não encontrados (rode `python train.py` ou copie a pasta `checkpoints`):\n  "
                + "\n  ".join(absent)
            )


def run_pipeline(argv):
    from pipeline import main

    return main(argv)


class Launcher(Slotted):
    """Só pede dependências e checkpoints quando vai de fato carregar modelos."""

    __slots__ = ("argv", "dependencies", "models", "entry")

    def __init__(self, argv, dependencies=None, models=None, entry=run_pipeline):
        self.argv = argv
        self.dependencies = dependencies or DependencyCheck()
        self.models = models or ModelCheck()
        self.entry = entry

    def run(self):
        if not any(arg in HELP_FLAGS for arg in self.argv) and '--metrics-only' not in self.argv:
            self.dependencies.ensure()
            self.models.ensure(self.argv)
        return self.entry(self.argv)


if __name__ == "__main__":
    raise SystemExit(Launcher(sys.argv[1:]).run())
