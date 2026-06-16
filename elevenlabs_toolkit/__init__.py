from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "elevenlabs_toolkit"
if _SRC_PACKAGE.is_dir():
    __path__.insert(0, str(_SRC_PACKAGE))
    _src_init = _SRC_PACKAGE / "__init__.py"
    if _src_init.is_file():
        exec(compile(_src_init.read_text(encoding="utf-8-sig"), str(_src_init), "exec"), globals())
