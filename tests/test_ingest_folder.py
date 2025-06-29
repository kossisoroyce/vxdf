from pathlib import Path

from vxdf.ingest import convert
from vxdf.reader import VXDFReader


def test_convert_directory(tmp_path: Path) -> None:
    # create sample txt files
    d = tmp_path / "docs"
    d.mkdir()
    file1 = d / "a.txt"
    file1.write_text("hello world\n")
    file2 = d / "b.txt"
    file2.write_text("goodbye\n")

    out = tmp_path / "out.vxdf"
    convert(d, out)

    r = VXDFReader(str(out))
    assert len(r.offset_index) == 2
    assert r.get_chunk("a-line-0")["text"] == "hello world"
    r.file.close()
