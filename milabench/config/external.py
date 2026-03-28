import os
import subprocess



milabench = os.path.dirname(__file__)
benchmarks = os.path.join(milabench, "..", "..", "benchmarks")
extern_location = os.path.join(benchmarks, "_extern")


def resolve_extern_definition(pack):
    os.makedirs(extern_location, exist_ok=True)

    if isinstance(pack['definition'], str):
        return

    url = pack["definition"]["git"]
    branch = pack["definition"].get("branch", "main")

    name = os.path.splitext(os.path.basename(url))[0]
    definition_path = os.path.join(extern_location, name)
    
    if not os.path.exists(definition_path):
        cwd = extern_location
        cmd = ["git", "clone", "--recurse-submodules", "-j8", "-b", branch, url]

        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # gives you strings instead of bytes
        )

        if result.returncode != 0:
            raise RuntimeError("Could not resolve definition of benchmark")

    # Done

    pack["definition"] = os.path.join(extern_location, name)