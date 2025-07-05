import os
import sys
import subprocess
from pathlib import Path

import pytest

from vxdf.reader import VXDFReader

# Ensure the local source is preferred over any installed version.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
CLI_ENTRY = [sys.executable, "-m", "vxdf"]


def _vxdf_count(path: Path) -> int:
    """Return number of chunks in *path*."""
    with VXDFReader(str(path)) as r:
        return len(r.offset_index)



def _run_cli(args, cwd, *, input_data=None):
    """Helper that runs the VXDF CLI and returns CompletedProcess."""
    env = os.environ.copy()
    # Prepend project root to PYTHONPATH to ensure the local source is used.
    env["PYTHONPATH"] = f"{_PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    if input_data is not None:
        # For piping tests, handle binary I/O
        return subprocess.run(
            CLI_ENTRY + args,
            cwd=cwd,
            input=input_data.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=env,
        )
    # For other tests, maintain original behavior (text output, stdout/stderr merged)
    return subprocess.run(
        CLI_ENTRY + args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def test_cli_incremental_update(tmp_path: Path):
    """Round-trip: convert folder, append one doc, ensure count increases."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("first document")
    (data_dir / "b.txt").write_text("second doc")

    out = tmp_path / "docs.vxdf"

    # Initial convert should create 2 chunks
    res = _run_cli(["convert", str(data_dir), str(out)], cwd=str(tmp_path))
    assert res.returncode == 0, res.stdout
    assert _vxdf_count(out) == 2

    # Append a third file using the CLI alias "append"
    (data_dir / "c.txt").write_text("third doc")
    res2 = _run_cli(["append", str(out), str(data_dir / "c.txt")], cwd=str(tmp_path))
    assert res2.returncode == 0, res2.stdout

    # Verify VXDF now has 3 chunks
    assert _vxdf_count(out) == 3



def test_cli_merge_split(tmp_path: Path):
    """End-to-end check for `vxdf merge` and `vxdf split`."""
    # Prepare two tiny corpora (one file each)
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir(); d2.mkdir()
    (d1 / "x.txt").write_text("alpha")
    (d2 / "y.txt").write_text("beta")

    f1 = tmp_path / "one.vxdf"
    f2 = tmp_path / "two.vxdf"
    assert _run_cli(["convert", str(d1), str(f1)], cwd=tmp_path).returncode == 0
    assert _run_cli(["convert", str(d2), str(f2)], cwd=tmp_path).returncode == 0

    merged = tmp_path / "merged.vxdf"
    res = _run_cli(["merge", str(merged), str(f1), str(f2)], cwd=tmp_path)
    assert res.returncode == 0, res.stdout
    assert _vxdf_count(merged) == 2

    # Split into 1-chunk parts
    res2 = _run_cli(["split", str(merged), "--chunks", "1"], cwd=tmp_path)
    assert res2.returncode == 0, res2.stdout

    part1 = tmp_path / "merged-part01.vxdf"
    part2 = tmp_path / "merged-part02.vxdf"
    assert part1.is_file() and part2.is_file(), "Split parts not created"
    assert _vxdf_count(part1) == 1 and _vxdf_count(part2) == 1


@pytest.mark.skip(reason="Cloud path handling requires configured AWS/GCP creds")
@pytest.mark.parametrize("url", [
    "https://raw.githubusercontent.com/vxdf-ai/test-assets/main/sample.txt",
    "s3://vxdf-test/sample.txt",
])
def test_cli_remote_ingest(url, tmp_path):
    """Ensure remote file paths are accepted."""
    out = tmp_path / "remote.vxdf"
    res = _run_cli(["convert", url, str(out)], cwd=str(tmp_path))
    assert res.returncode == 0, res.stdout


def test_cli_stdin_stdout(tmp_path: Path):
    """Pipe a small text into vxdf convert - output to stdout."""
    input_text = "hello from stdin"
    # Use --model all-MiniLM-L6-v2 to avoid OpenAI dependency in this test
    # Use --no-summary to keep it simple
    cmd = [
        "convert",
        "-",
        "-",
        "--model",
        "all-MiniLM-L6-v2",
        "--no-summary",
        "--no-progress",
    ]
    res = _run_cli(cmd, cwd=str(tmp_path), input_data=input_text)
    assert res.returncode == 0, res.stderr.decode()
    assert len(res.stdout) > 500  # Check that we got some binary output

    # Verify the output is a valid VXDF file by writing to disk and reading it
    out_vxdf = tmp_path / "output.vxdf"
    out_vxdf.write_bytes(res.stdout)

    reader = VXDFReader(str(out_vxdf))
    assert len(reader.offset_index) == 1, "Expected one chunk in the output VXDF"

    # Get the first (and only) chunk ID, making the test robust to ID generation logic
    try:
        first_id = next(iter(reader.offset_index.keys()))
    except StopIteration:
        pytest.fail("VXDF file was created but contains no chunks.")

    chunk = reader.get_chunk(first_id)
    assert chunk["text"] == input_text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific path test")
def test_windows_non_utf8_path(tmp_path):
    """Placeholder for Windows non-UTF-8 path handling."""
    pass


@pytest.mark.skip(reason="progress bar/ETA visual check – manual")
def test_cli_progress_bar(tmp_path):
    """Ensure --no-progress suppresses bar for CI stability."""
    pass
