"""Every test file under tests/ is selected by a CI job -- and that job can actually run its tests.

    python -I tests/test_ci_registration.py

``.github/workflows/ci.yml`` uses ``scripts/list_python_tests.py`` to route ``tests/test_*.py``:
``tests/pytest-targets.txt`` lists pytest's targets; the remaining files run with ``python -I``.
``tests/semantic/`` goes to ``unittest discover`` and ``tests/*.cjs`` to ``node --test``.
This is the test-file counterpart of ``tests/test_ctest_registration.py``: it shares the Python selector
with the workflow and reads the semantic and viewer patterns from it. It fails on ways a test goes missing
without CI turning red --

* a file no job selects, because no glob matches its name: ``tests/thing_test.py`` matches neither
  ``test_*.py`` nor ``test*.py``;
* a file listed for pytest that keeps its only checks under ``if __name__ == "__main__":`` -- pytest
  imports it, collects nothing, and the job stays green because the other files supply items;
* a file the loop runs with ``python -I`` that only *mentions* ``__main__``, or whose guard is inverted
  (``if __name__ != "__main__":``) and runs nothing;
* a semantic file with only module-level ``test*`` functions: ``unittest discover`` collects nothing
  there, even though pytest would; and
* a nested ``tests/*/test_*.cjs`` file that ``fnmatch`` matches across ``/`` but the non-recursive
  ``node --test tests/test_*.cjs`` command never runs.

It also runs the ``check`` loop itself, taken from the workflow, under bash on a synthetic tree: the loop has
to run every file and then fail, since stopping at the first failure hides the files sorted after it.

No game files, build tools or pytest are needed. The loop check needs bash (Git for Windows' on Windows)
and skips without it, except on a GitHub runner.
"""
import ast
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(".github", "workflows", "ci.yml")
SELECTOR = "scripts/list_python_tests.py"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from list_python_tests import PYTEST_TARGETS, SEMANTIC_DIR, python_test_files, routes_for  # noqa: E402

# What a file has to be called to claim it is a test. Any of these that no job selects is a bug, which is why
# the suffix form (tests/thing_test.py) counts even though no job's glob matches it today.
TEST_FILE = re.compile(r"^(?:test.*\.py|.*_test\.py|test.*\.cjs)$")


def read(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as f:
        return f.read()


def ci_rules(root):
    """(standalone files, pytest modules, semantic directory, viewer glob)."""
    text = read(root, WORKFLOW)

    def find(pattern, what):
        match = re.search(pattern, text, re.M)
        if not match:
            raise AssertionError(
                f"ci.yml no longer spells {what} the way tests/test_ci_registration.py expects it to; "
                "update this test together with the workflow")
        return match.groups()

    # scripts/run_semantic_tests.py runs the files unittest discover would find, one process each
    (semantic,) = find(r"(?:python\s+-m\s+unittest\s+discover\s+-s|python\s+-I\s+scripts/run_semantic_tests\.py)\s+(\S+)",
                       "the semantic suite's folder")
    (viewer,) = find(r"node\s+--test\s+(\S+)", "node --test")
    if re.search(r"unittest\s+discover[^\n]*-p\s", text):
        raise AssertionError("unittest discover now passes -p; this test assumes its default 'test*.py' pattern")
    if semantic.rstrip("/") != SEMANTIC_DIR:
        raise AssertionError(f"the selector prunes {SEMANTIC_DIR}, but ci.yml runs {semantic}")
    stdlib, pytest = routes_for(root)
    return set(stdlib), {target for target in pytest if "::" not in target}, SEMANTIC_DIR, viewer


STDLIB_STEP = "Tests (standard library only, no game data)"


def step_script(root, step_name):
    """The ``run: |`` block of the workflow step called ``step_name``, dedented, exactly as CI runs it."""
    lines = read(root, WORKFLOW).splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"- name: {step_name}":
            step_indent = len(line) - len(line.lstrip()) + 2
            for j in range(i + 1, len(lines)):
                text = lines[j]
                if text.strip() and len(text) - len(text.lstrip()) < step_indent:
                    break                          # the next step: this one has no run block
                if text.strip() == "run: |":
                    body = []
                    for k in range(j + 1, len(lines)):
                        if lines[k].strip() and len(lines[k]) - len(lines[k].lstrip()) <= step_indent:
                            break
                        body.append(lines[k])
                    indent = min(len(b) - len(b.lstrip()) for b in body if b.strip())
                    return "\n".join(b[indent:] for b in body).strip("\n") + "\n"
    raise AssertionError(f"ci.yml has no step named {step_name!r} with a `run: |` block; update "
                         "tests/test_ci_registration.py together with the workflow")


def git_bash():
    """The bash GitHub's `shell: bash` uses: Git for Windows' on Windows (not WSL's), PATH's elsewhere."""
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    if git:
        folder = os.path.dirname(os.path.realpath(git))
        for _ in range(3):
            folder = os.path.dirname(folder)
            for candidate in (os.path.join(folder, "bin", "bash.exe"), os.path.join(folder, "usr", "bin", "bash.exe")):
                if os.path.isfile(candidate):
                    return candidate
    return None


def tracked_under_tests(root):
    try:
        result = subprocess.run(["git", "ls-files", "-z", "--", "tests"], cwd=root, capture_output=True, text=True)
        if result.returncode == 0:
            return [n for n in result.stdout.split("\0") if n]
    except OSError:
        pass
    names = []                                     # no git: walk, as tests/test_docs_links.py does
    for base, dirs, files in os.walk(os.path.join(root, "tests")):
        dirs[:] = [d for d in dirs if d not in (".git", "third_party", "__pycache__")]
        for name in files:
            names.append(os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/"))
    return names


def claimed_test_files(root):
    """Tracked files under tests/ that claim to be tests: by name, or because they define one."""
    found = []
    for rel in tracked_under_tests(root):
        name = rel.rsplit("/", 1)[-1]
        if TEST_FILE.match(name):
            found.append(rel)
        elif name.endswith(".py"):
            try:
                if defines_a_test(read(root, rel)):
                    found.append(rel)              # a test under a name no job's glob matches
            except SyntaxError:
                pass
    return sorted(found)


def matches_nonrecursive_glob(rel, pattern):
    """Match the way the shell and ``node --test`` do: ``*`` does not cross ``/``.

    ``fnmatch`` lets ``tests/test_*.cjs`` match ``tests/sub/test_x.cjs``, but the workflow command
    only expands the top level, so those files would be selected by this test and never run.
    """
    directory, _, name = rel.rpartition("/")
    pattern_dir, _, pattern_name = pattern.rpartition("/")
    return directory == pattern_dir and fnmatch.fnmatchcase(name, pattern_name)


def job_of(root, rel, rules):
    """Which job runs ``rel``: 'stdlib', 'pytest', 'semantic', 'viewer', or None for no job at all."""
    stdlib, pytest, semantic, viewer = rules
    if rel.startswith(semantic + "/"):
        directory, _, name = rel.rpartition("/")
        # unittest discovery only descends into packages beneath its start directory.
        while directory != semantic:
            if not os.path.isfile(os.path.join(root, directory, "__init__.py")):
                return None
            directory = directory.rpartition("/")[0]
        return "semantic" if fnmatch.fnmatchcase(name, "test*.py") else None
    if matches_nonrecursive_glob(rel, viewer):
        return "viewer"
    if rel in pytest:
        return "pytest"
    if rel in stdlib:
        return "stdlib"
    return None


DOCS_ONLY_STEP = "Check documentation when only documentation changed"
# The tests that read only the documents. `check` is skipped for a docs-only pull request, so the docs-only step
# of `change-scope` is the one place they run for it.
DOCUMENT_TESTS = ("tests/test_docs_*.py", "tests/test_i18n_index.py")


def docs_only_patterns(root):
    """The test files or globs the docs-only step runs, exactly as the workflow spells them."""
    text = read(root, WORKFLOW)
    start = text.find("- name: " + DOCS_ONLY_STEP)
    if start < 0:
        raise AssertionError(
            f"ci.yml no longer has the step '{DOCS_ONLY_STEP}'; update tests/test_ci_registration.py together "
            "with the workflow")
    # The step ends at the next line indented no deeper than its own "- name:": another step, or another job.
    end = re.compile(r"^ {0,6}\S", re.M).search(text, text.index("\n", start))
    # A commented-out command is the usual way a test stops running: shell comments don't count. (`#` starts one
    # at the beginning of a line or after a space, as in bash.)
    lines = [re.sub(r"(^|\s)#.*", "", line) for line in text[start:end.start() if end else len(text)].splitlines()]
    return re.findall(r"tests/[\w*./-]+\.py", "\n".join(lines))


def document_tests_not_run_for_docs_only(root, files):
    """Document tests the docs-only step does not run, and names in the step that match no tracked file."""
    patterns = docs_only_patterns(root)
    wanted = [f for f in files if any(matches_nonrecursive_glob(f, p) for p in DOCUMENT_TESTS)]
    missing = [f for f in wanted if not any(matches_nonrecursive_glob(f, p) for p in patterns)]
    stale = [p for p in patterns if not any(matches_nonrecursive_glob(f, p) for f in files)]
    return missing, stale


def defines_a_test(source, require_test_case=False):
    """Would the runner find a test here? TestCase subclasses (through local bases) and test* functions.

    ``unittest discover`` only collects TestCase subclasses, so the semantic job passes
    ``require_test_case=True``: a module with only ``def test_x(): ...`` runs zero cases there even
    though pytest would collect it.

    A TestCase runs the ``test*`` methods it inherits as well as its own (the loader lists them with ``dir()``),
    so a TestCase with none of its own counts when a class of this module that it derives from defines one: the
    mixin of ``tests/semantic/test_sub_co_same_operand.py``, shared by one TestCase per wave size.
    """
    tree = ast.parse(source)
    bases, own_tests, case_aliases = {}, {}, set()
    # Recognize bare aliases introduced by a direct import; this is not a general name resolver.
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases[node.name] = [
                "TestCase" if isinstance(b, ast.Name) and b.id in case_aliases else ast.unparse(b).rsplit(".", 1)[-1]
                for b in node.bases
            ]
            own_tests[node.name] = any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                                       and item.name.startswith("test") for item in node.body)
            case_aliases.discard(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for item in node.names:
                name = item.asname or item.name.split(".", 1)[0]
                if (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "unittest"
                        and item.name == "TestCase" and item.asname):
                    case_aliases.add(name)
                else:
                    case_aliases.discard(name)
        elif isinstance(node, ast.Assign):
            case_aliases.difference_update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
            # A bare annotation does not change the imported binding.
            case_aliases.discard(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            case_aliases.discard(node.name)

    def is_test_case(name):
        seen, stack = set(), [name]
        while stack:
            for base in bases.get(stack.pop(), []):
                if base in seen:
                    continue
                seen.add(base)
                if base == "TestCase":
                    return True
                stack.append(base)
        return False

    def local_ancestors(name):
        """The classes of this module `name` derives from, directly or through other classes of this module."""
        seen, stack = [], [name]
        while stack:
            for base in bases[stack.pop()]:
                if base in bases and base not in seen:
                    seen.append(base)
                    stack.append(base)
        return seen

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and is_test_case(node.name):
            if any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test")
                   for item in node.body):
                return True
            # the test* methods of the classes of this module it derives from are its tests too
            ancestors = local_ancestors(node.name)
            if any(own_tests[name] for name in ancestors):
                return True
            # a mixin imported from another module (tests/semantic/f16_unary_cases.py's UnaryF16) brings its tests,
            # to the class that names it and to the classes of this module that derive from that one
            if any(b not in bases and b not in ("TestCase", "object")
                   for name in [node.name] + ancestors for b in bases[name]):
                return True
        if not require_test_case and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name.startswith("test"):
            return True
    return False


def semantic_job_text(workflow):
    """The `semantic-tests:` job of the workflow, up to the next job."""
    match = re.search(r"^  semantic-tests:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)", workflow, re.M | re.S)
    if not match:
        raise AssertionError("ci.yml no longer has a `semantic-tests:` job; update tests/test_ci_registration.py")
    return match.group(1)


def spirv_gated_semantic_files(root, files):
    """tests/semantic files that skip when spirv-as / spirv-val are not installed."""
    gated = []
    for rel in files:
        if not rel.startswith("tests/semantic/") or not rel.endswith(".py"):
            continue
        source = read(root, rel)
        if re.search(r"which\(\s*[\"']spirv-(?:as|val)[\"']", source) and re.search(r"skip(?:Unless|Test|If)\b", source):
            gated.append(rel)
    return gated


def spirv_tools_problem(workflow, gated):
    """Why the semantic job cannot run `gated` tests, or None: it needs an apt install of spirv-tools and the
    LEONIDA_REQUIRE_SPIRV_TOOLS=1 switch that makes a missing tool fail."""
    if not gated:
        return None
    job = semantic_job_text(workflow)
    problems = []
    if not re.search(r"apt-get\s+install[^\n]*\bspirv-tools\b", job):
        problems.append("it does not `apt-get install` spirv-tools")
    if not re.search(r"LEONIDA_REQUIRE_SPIRV_TOOLS:[^\n]*['\"]1['\"]", job):
        problems.append("it never sets LEONIDA_REQUIRE_SPIRV_TOOLS to 1")
    return "; ".join(problems) or None


def spirv_using_stdlib_files(root, files, rules):
    """Files the standard-library loop runs that call spirv-as / spirv-val only when they are installed."""
    return [rel for rel in files if job_of(root, rel, rules) == "stdlib"
            and re.search(r"which\(\s*[\"']spirv-(?:as|val)[\"']", read(root, rel))]


def stdlib_loop_jobs_with_spirv_tools(workflow):
    """Jobs that `apt-get install` spirv-tools and run the standard-library loop (`python -I "$t"`)."""
    jobs = re.findall(r"^  ([A-Za-z0-9_-]+):\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)", workflow, re.M | re.S)
    return [name for name, job in jobs if re.search(r'python\s+-I\s+"\$t"', job)
            and re.search(r"apt-get\s+install[^\n]*\bspirv-tools\b", job)]


def has_main_guard(source):
    """A real ``if __name__ == "__main__":`` block that runs something.

    A comment or docstring mentioning ``__main__`` is not enough, and neither is an inverted guard
    such as ``if __name__ != "__main__": unittest.main()``: ``python -I`` imports the file, the
    comparison is true, ``unittest.main()`` is skipped, and the process still exits 0.
    """
    def is_name(node):
        return isinstance(node, ast.Name) and node.id == "__name__"

    def is_main(node):
        return isinstance(node, ast.Constant) and node.value == "__main__"

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        left, right = test.left, test.comparators[0]
        if not ((is_name(left) and is_main(right)) or (is_main(left) and is_name(right))):
            continue
        if any(isinstance(item, ast.Call) for item in ast.walk(node)):
            return True
    return False


class RepositoryRegistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = ci_rules(ROOT)
        cls.files = claimed_test_files(ROOT)

    def test_the_scan_sees_the_repository(self):
        # A glob that stops matching, or a git call that fails, must not turn the checks below into no-ops.
        self.assertGreaterEqual(len(self.files), 300, f"only {len(self.files)} test files found: {self.files[:5]}")
        self.assertTrue(any(f.startswith("tests/semantic/") for f in self.files))

    def test_the_scan_falls_back_when_git_is_missing(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("git")):
            files = tracked_under_tests(ROOT)
            self.assertGreaterEqual(len(files), 300)
            self.assertTrue(any(f.startswith("tests/semantic/") for f in files))

    def test_every_test_file_is_selected_by_a_job(self):
        unselected = [f for f in self.files if job_of(ROOT, f, self.rules) is None]
        self.assertEqual(unselected, [],
                         "no CI job runs these test files: " + ", ".join(unselected) + "\n"
                         "Rename them to tests/test_*.py (or tests/test_*.cjs), put them under tests/semantic/ "
                         "as test*.py with an __init__.py in each nested directory, or add the pattern to ci.yml.")

    def test_whole_file_routes_partition_nonsemantic_python_tests(self):
        standalone, targets = routes_for(ROOT)
        modules = [t for t in targets if "::" not in t]
        self.assertEqual(sorted(standalone + modules), python_test_files(ROOT))
        self.assertIn("tests/test_tex_headers.py", standalone)
        self.assertIn("tests/test_tex_headers.py::TestTextureDescriptorParsing::"
                      "test_stdlib_qword_scanner_edge_cases_and_parity", targets)

    def test_selector_writes_lf_even_with_crlf_stdout(self):
        # Emulate Python's Windows stdout on every host and capture bytes without newline normalization.
        wrapper = ("import runpy, sys; "
                   "sys.stdout.reconfigure(newline='\\r\\n'); "
                   "runpy.run_path(sys.argv.pop(1), run_name='__main__')")
        for runner, targets in zip(("stdlib", "pytest"), routes_for(ROOT)):
            with self.subTest(runner=runner):
                result = subprocess.run([sys.executable, "-I", "-c", wrapper, SELECTOR, runner],
                                        cwd=ROOT, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn(b"\r", result.stdout)
                self.assertEqual(result.stdout.decode("utf-8").splitlines(), targets)

    def test_manifest_rejects_stale_and_overlapping_targets(self):
        file = "tests/test_example.py"
        node = file + "::test_parity"
        cases = (
            ("tests/test_gone.py", "invalid or missing"),
            ("tests/semantic/test_hidden.py", "invalid or missing"),
            (file + "::", "invalid or missing"),
            (f"{file}\n{file}", "duplicate or overlapping"),
            (f"{node}\n{node}", "duplicate or overlapping"),
            (f"{file}\n{node}", "duplicate or overlapping"),
            (f"{node}\n{file}", "duplicate or overlapping"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, SEMANTIC_DIR))
            for path in (file, "tests/semantic/test_hidden.py"):
                with open(os.path.join(tmp, path), "w", encoding="utf-8", newline="\n"):
                    pass
            for contents, error in cases:
                with self.subTest(contents=contents):
                    with open(os.path.join(tmp, PYTEST_TARGETS), "w", encoding="utf-8", newline="\n") as f:
                        f.write(contents + "\n")
                    with self.assertRaisesRegex(ValueError, error):
                        routes_for(tmp)

    def test_direct_testcase_import_alias(self):
        import_alias = "from unittest import TestCase as Parent\n"
        test_class = "class Example({base}):\n    def test_one(self): pass\n"
        direct = test_class.format(base="Parent")
        cases = (
            ("direct", import_alias + direct, True),
            ("local subclass", import_alias + "class Middle(Parent): pass\n"
             + test_class.format(base="Middle"), True),
            ("empty", import_alias + "class Example(Parent): pass\n", False),
            ("unrelated import", "from types import SimpleNamespace as Parent\n" + direct, False),
            ("relative import", "from .unittest import TestCase as Parent\n" + direct, False),
            ("qualified name", import_alias + test_class.format(base="other.Parent"), False),
            ("reassigned", import_alias + "Parent = object\n" + direct, False),
            ("annotated assignment before class", import_alias + "Parent: type = object\n" + direct, False),
            ("annotation only", import_alias + "Parent: type\n" + direct, True),
            ("annotated assignment after class", import_alias + direct + "Parent: type = object\n", True),
            ("local shadow", import_alias + "class Parent: pass\n" + direct, False),
            ("reimported", import_alias + "from types import SimpleNamespace as Parent\n" + direct, False),
        )
        for kind, source, expected in cases:
            with self.subTest(kind=kind):
                self.assertEqual(defines_a_test(source, require_test_case=True), expected)

    def test_inherited_tests_count_where_unittest_discover_collects_them(self):
        # The expectations are the runner's: unittest discover is run on these modules, and the scan has to find a
        # test in exactly the ones it collects one from.
        mixin = "class Mixin:\n    def test_one(self): pass\n"
        cases = (
            ("mixin_above", "import unittest\n" + mixin + "class Example(Mixin, unittest.TestCase): pass\n", True),
            ("mixin_two_classes_up", "import unittest\n" + mixin + "class Middle(Mixin): pass\n"
             "class Example(Middle, unittest.TestCase): pass\n", True),
            ("mixin_beside_an_empty_case", "import unittest\n" + mixin + "class Base(unittest.TestCase): pass\n"
             "class Example(Mixin, Base): pass\n", True),
            ("imported_mixin", "import unittest\nfrom shared_cases import Cases\n"
             "class Example(Cases, unittest.TestCase): pass\n", True),
            ("imported_mixin_through_a_local_class", "import unittest\nfrom shared_cases import Cases\n"
             "class Middle(Cases): pass\nclass Example(Middle, unittest.TestCase): pass\n", True),
            ("base_without_tests", "import unittest\nclass Helper:\n    def helper(self): pass\n"
             "class Example(Helper, unittest.TestCase): pass\n", False),
            ("mixin_no_case_derives_from", "import unittest\n" + mixin
             + "class Example(unittest.TestCase): pass\n", False),
            ("mixin_without_a_case", mixin + "class Example(Mixin): pass\n", False),
        )
        with tempfile.TemporaryDirectory() as tmp:
            sources = [("shared_cases.py", "class Cases:\n    def test_one(self): pass\n")]
            sources += [(f"test_{kind}.py", source) for kind, source, _ in cases]
            for name, source in sources:
                with open(os.path.join(tmp, name), "w", encoding="utf-8", newline="\n") as f:
                    f.write(source)
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", ".", "-v"],
                                    cwd=tmp, capture_output=True, text=True, timeout=30, check=True)
        for kind, source, expected in cases:
            with self.subTest(kind=kind):
                self.assertEqual(f"(test_{kind}.Example" in result.stderr, expected, result.stderr)
                self.assertEqual(defines_a_test(source, require_test_case=True), expected)
        self.assertIn("Ran 5 tests", result.stderr)

    def test_every_selected_file_exposes_a_test_to_its_runner(self):
        # Selection is not execution: pytest imports a file it is handed and reports zero items without
        # failing, and unittest discover ignores a module with no TestCase.
        silent = []
        for rel in self.files:
            job = job_of(ROOT, rel, self.rules)
            if job not in ("pytest", "semantic") or not rel.endswith(".py"):
                continue
            if not defines_a_test(read(ROOT, rel), require_test_case=(job == "semantic")):
                silent.append((job, rel))
        self.assertEqual(silent, [], "selected by CI, but the runner would collect no test:\n" + "\n".join(
            f"  {rel}: {job} imports it and collects nothing, so the job stays green. Give it a unittest.TestCase "
             f"or a module-level test* function, or correct its routing in {PYTEST_TARGETS}."
            for job, rel in silent))

    def test_every_file_the_loop_runs_has_a_main_guard(self):
        # A standalone file with no entry point exits successfully without running anything.
        guardless = [f for f in self.files
                     if job_of(ROOT, f, self.rules) == "stdlib" and not has_main_guard(read(ROOT, f))]
        self.assertEqual(
            guardless, [], "the check loop runs these with python -I, which executes no test:\n"
            + "\n".join(f"  {f}: no `if __name__ == \"__main__\":` block" for f in guardless)
            + f"\nIf these are pytest files, list them in {PYTEST_TARGETS}.")

    def test_the_semantic_job_installs_what_its_spirv_gated_tests_need(self):
        # A test that skips without spirv-as / spirv-val runs only where something installed them. The runner images
        # have neither, so a job that skips instead of installing reports green for tests that never ran.
        gated = spirv_gated_semantic_files(ROOT, self.files)
        self.assertGreaterEqual(len(gated), 3, f"the scan found only {gated}")
        for expected in ("tests/semantic/test_fma_khr.py", "tests/semantic/test_bary_fallback.py",
                         "tests/semantic/test_image_coherent.py"):
            self.assertIn(expected, gated)
        problem = spirv_tools_problem(read(ROOT, WORKFLOW), gated)
        self.assertIsNone(problem, "the `semantic-tests` job in ci.yml runs tests that skip without SPIRV-Tools, but " +
                          str(problem) + ".\nInstall spirv-tools on the Linux leg and set LEONIDA_REQUIRE_SPIRV_TOOLS "
                          "to '1' there (see tests/semantic/test_spirv_tools_required.py).")

    def test_a_job_runs_the_spirv_using_stdlib_files_with_spirv_tools(self):
        # The check job's Windows runner has no SPIRV-Tools, so these files skip their spirv-val checks there.
        used = spirv_using_stdlib_files(ROOT, self.files, self.rules)
        self.assertIn("tests/test_split_big_switch.py", used)
        workflow = read(ROOT, WORKFLOW)
        self.assertEqual(stdlib_loop_jobs_with_spirv_tools(workflow.replace("spirv-tools", "curl")), [])
        self.assertTrue(stdlib_loop_jobs_with_spirv_tools(workflow), "no job in ci.yml installs SPIRV-Tools and "
                        "runs the standard-library loop, so the spirv-val checks of these files never run:\n  " +
                        "\n  ".join(used))

    def test_the_spirv_tools_check_flags_a_job_that_installs_nothing(self):
        workflow = read(ROOT, WORKFLOW)
        gated = ["tests/semantic/test_fma_khr.py"]
        self.assertIsNone(spirv_tools_problem(workflow, gated))
        self.assertIsNone(spirv_tools_problem(workflow, []))
        no_install = workflow.replace("sudo apt-get install -y --no-install-recommends spirv-tools\n",
                                      "sudo apt-get install -y --no-install-recommends curl\n", 1)
        self.assertNotEqual(no_install, workflow)
        self.assertIn("apt-get", no_install)
        # the translator-tests job above installs spirv-tools too; only the semantic job's own text may count
        self.assertIn("does not `apt-get install` spirv-tools", spirv_tools_problem(no_install, gated) or "")
        no_switch = workflow.replace("LEONIDA_REQUIRE_SPIRV_TOOLS: ${{ runner.os == 'Linux' && '1' || '0' }}",
                                     "LEONIDA_REQUIRE_SPIRV_TOOLS: ${{ runner.os == 'Linux' && 'yes' || 'no' }}")
        self.assertNotEqual(no_switch, workflow)
        self.assertIn("never sets LEONIDA_REQUIRE_SPIRV_TOOLS", spirv_tools_problem(no_switch, gated) or "")

    def test_a_docs_only_pull_request_runs_every_document_test(self):
        # `check` is skipped when only documents changed, so a document test left out of the docs-only step never
        # sees the pull request that breaks it: the pull request is green and `main` fails after the merge.
        missing, stale = document_tests_not_run_for_docs_only(ROOT, self.files)
        self.assertEqual(missing, [], f"the '{DOCS_ONLY_STEP}' step in ci.yml does not run: " + ", ".join(missing))
        self.assertEqual(stale, [], "that step names test files that do not exist: " + ", ".join(stale))
        # The step runs them with `python3 -I`: each has to be a standard-library script with a main guard.
        ran = [f for f in self.files if any(matches_nonrecursive_glob(f, p) for p in docs_only_patterns(ROOT))]
        self.assertGreaterEqual(len(ran), 5, ran)
        for rel in ran:
            with self.subTest(file=rel):
                self.assertEqual(job_of(ROOT, rel, self.rules), "stdlib")
                self.assertTrue(has_main_guard(read(ROOT, rel)))

    def test_the_scan_flags_a_document_test_the_docs_only_step_leaves_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".github", "workflows"))
            files = ["tests/test_docs_links.py", "tests/test_docs_design_rows.py", "tests/test_i18n_index.py",
                     "tests/test_other.py"]

            def workflow(run_lines):
                with open(os.path.join(tmp, WORKFLOW), "w", encoding="utf-8") as f:
                    f.write("jobs:\n  change-scope:\n    steps:\n"
                            f"      - name: {DOCS_ONLY_STEP}\n        run: |\n"
                            + "".join(f"          {line}\n" for line in run_lines)
                            + "  check:\n    steps:\n      - run: python -I tests/test_other.py\n")

            # The step as it was: two files by name. The design-rows test is a document test it never ran.
            workflow(["python3 -I tests/test_docs_links.py", "python3 -I tests/test_i18n_index.py"])
            self.assertEqual(document_tests_not_run_for_docs_only(tmp, files),
                             (["tests/test_docs_design_rows.py"], []))
            workflow(["for t in tests/test_docs_*.py tests/test_i18n_index.py; do", "  python3 -I \"$t\" || exit 1",
                      "done"])
            self.assertEqual(document_tests_not_run_for_docs_only(tmp, files), ([], []))
            # A name that matches nothing: bash would hand the literal pattern to python3 and the step would fail.
            workflow(["python3 -I tests/test_docs_links.py", "python3 -I tests/test_docs_gone.py"])
            self.assertEqual(document_tests_not_run_for_docs_only(tmp, files),
                             (["tests/test_docs_design_rows.py", "tests/test_i18n_index.py"],
                              ["tests/test_docs_gone.py"]))
            # The step is read up to the next step or job, not to the end of the file.
            self.assertNotIn("tests/test_other.py", docs_only_patterns(tmp))
            # A commented-out command does not run: neither a whole line nor the rest of a line after ` #`.
            workflow(["# for t in tests/test_docs_*.py tests/test_i18n_index.py; do", "echo skipped",
                      "python3 -I tests/test_docs_links.py  # python3 -I tests/test_i18n_index.py"])
            self.assertEqual(docs_only_patterns(tmp), ["tests/test_docs_links.py"])
            self.assertEqual(document_tests_not_run_for_docs_only(tmp, files),
                             (["tests/test_docs_design_rows.py", "tests/test_i18n_index.py"], []))

    def test_the_scan_flags_an_unlisted_file_and_a_file_pytest_cannot_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".github", "workflows"))
            os.makedirs(os.path.join(tmp, "tests", "semantic"))
            os.makedirs(os.path.join(tmp, "tests", "test_nested_dir"))
            with open(os.path.join(tmp, WORKFLOW), "w", encoding="utf-8") as f:
                f.write(read(ROOT, WORKFLOW))
            def write(name, text):
                with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
                    f.write(text)
            write("tests/test_plain.py", "import unittest\n"
                  "class T(unittest.TestCase):\n    def test_ok(self):\n        pass\n"
                  "if __name__ == \"__main__\":\n    unittest.main()\n")
            # The manifest sends this to pytest, but it has no collectable tests.
            write(PYTEST_TARGETS, "tests/test_numpy_probe.py\n")
            write("tests/test_numpy_probe.py", "HAVE_NUMPY = False\n"
                  "def check():\n    assert 1 + 1 == 2\n"
                  "if __name__ == \"__main__\":\n    check()\n")
            write("tests/left_out_test.py", "import unittest\n"
                  "class T(unittest.TestCase):\n    def test_ok(self):\n        pass\n")
            write("tests/semantic/test_case.py", "import unittest\n"
                  "class T(unittest.TestCase):\n    def test_ok(self):\n        pass\n")
            write("tests/semantic/pytest_style.py", "def test_looks_like_a_test():\n    pass\n")
            # unittest discover ignores module-level test functions, so the semantic job would run zero.
            write("tests/semantic/test_function_only.py", "def test_passes():\n    pass\n")
            # The inverted guard passes a substring test but runs nothing under python -I.
            write("tests/test_inequality_guard.py", "import unittest\n"
                  "class T(unittest.TestCase):\n    def test_ok(self):\n        pass\n"
                  "if __name__ != \"__main__\":\n    unittest.main()\n")
            # fnmatch would match this against tests/test_*.cjs, but node --test does not recurse.
            write("tests/test_nested_dir/test_nested.cjs", "const test = require('node:test');\n")

            rules = ci_rules(tmp)
            found = claimed_test_files(tmp)
            selected = {rel: job_of(tmp, rel, rules) for rel in found}

            self.assertEqual(selected["tests/left_out_test.py"], None)
            self.assertEqual(selected["tests/semantic/pytest_style.py"], None)
            self.assertEqual(selected["tests/test_numpy_probe.py"], "pytest")
            self.assertFalse(defines_a_test(read(tmp, "tests/test_numpy_probe.py")))
            self.assertTrue(defines_a_test(read(tmp, "tests/test_plain.py")))
            self.assertTrue(has_main_guard(read(tmp, "tests/test_plain.py")))
            self.assertEqual(selected["tests/semantic/test_case.py"], "semantic")
            self.assertEqual(selected["tests/semantic/test_function_only.py"], "semantic")
            self.assertFalse(defines_a_test(read(tmp, "tests/semantic/test_function_only.py"),
                                            require_test_case=True))
            self.assertFalse(has_main_guard(read(tmp, "tests/test_inequality_guard.py")))
            self.assertEqual(selected["tests/test_nested_dir/test_nested.cjs"], None)

    def test_semantic_selection_matches_discovery_through_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rules = ci_rules(ROOT)
            semantic = rules[2]
            cases = ("test_root.py", "packaged/test_packaged.py", "bare/test_bare.py",
                     "bare/leaf/test_bare_ancestor.py")
            for case in cases:
                path = os.path.join(tmp, semantic, case)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("import unittest\nclass Case(unittest.TestCase):\n    def test_ok(self):\n        pass\n")
            for package in ("packaged", "bare/leaf"):
                with open(os.path.join(tmp, semantic, package, "__init__.py"), "w", encoding="utf-8", newline="\n") as f:
                    f.write("# Package for unittest discovery.\n")

            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", semantic, "-v"],
                                    cwd=tmp, capture_output=True, text=True, timeout=15, check=True)
            for case in cases:
                with self.subTest(case=case):
                    module = case.removesuffix(".py").replace("/", ".")
                    discovered = f"({module}.Case.test_ok)" in result.stderr
                    self.assertEqual(job_of(tmp, f"{semantic}/{case}", rules) == "semantic", discovered, result.stderr)
            self.assertIn("Ran 2 tests", result.stderr)


class StdlibLoopTest(unittest.TestCase):
    """The check job's loop runs every file and then fails, so one failure cannot hide the files after it."""

    def test_loop_runs_past_a_failure_and_names_every_failed_file(self):
        if not git_bash():
            # On a runner a missing bash would turn this check into a silent skip.
            self.assertFalse(os.environ.get("GITHUB_ACTIONS"), "no bash found on the CI runner")
            self.skipTest("needs bash (Git for Windows on Windows)")
        script = step_script(ROOT, STDLIB_STEP)
        failing = "import sys\nif __name__ == \"__main__\":\n    sys.exit(1)\n"
        passing = "import sys\nif __name__ == \"__main__\":\n    assert not sys.stdin.read()\n"
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "tests", "semantic"))
            os.makedirs(os.path.join(tmp, "scripts"))
            shutil.copyfile(os.path.join(ROOT, SELECTOR), os.path.join(tmp, SELECTOR))
            files = {
                "tests/test_a_fails.py": failing,
                "tests/test_b_passes.py": "HAVE_NUMPY = False\n" + passing,
                "tests/test_c_fails.py": failing,
                "tests/test_d_pytest.py": failing,  # skipped even though it could run as a script
                "tests/semantic/test_e.py": failing,                    # pruned: unittest discover runs it
                PYTEST_TARGETS: "tests/test_d_pytest.py\n",
            }
            for rel, text in files.items():
                with open(os.path.join(tmp, rel), "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
            with open(os.path.join(tmp, "step.sh"), "w", encoding="utf-8", newline="\n") as f:
                f.write(script)
            # The step calls `python`: a shim on PATH runs this interpreter (a POSIX box may only have `python3`).
            bindir = os.path.join(tmp, "bin")
            os.makedirs(bindir)
            shim = os.path.join(bindir, "python")
            with open(shim, "w", encoding="utf-8", newline="\n") as f:
                f.write("#!/bin/sh\nexec '%s' \"$@\"\n" % sys.executable.replace("'", "'\\''"))
            os.chmod(shim, 0o755)
            env = dict(os.environ, PATH=bindir + os.pathsep + os.environ.get("PATH", ""))
            # GitHub runs `shell: bash` as `bash --noprofile --norc -eo pipefail {0}`.
            result = subprocess.run([git_bash(), "--noprofile", "--norc", "-eo", "pipefail", "step.sh"], cwd=tmp,
                                    env=env, capture_output=True, text=True, timeout=120)
        out = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, out)
        ran = re.findall(r"^== (\S+)$", out, re.M)
        self.assertEqual(ran, ["tests/test_a_fails.py", "tests/test_b_passes.py", "tests/test_c_fails.py"], out)
        errors = re.findall(r"^::error file=(\S+)::", out, re.M)
        self.assertEqual(errors, ["tests/test_a_fails.py", "tests/test_c_fails.py"], out)

    def test_workflow_steps_use_the_shared_selector(self):
        for runner, names in (
            ("stdlib", (STDLIB_STEP, "Tests (standard library only, Linux)",
                        "Translator semantic tests and tests (standard library only, no game data)")),
            ("pytest", ("Translator tests", "Translator tests (pytest)")),
        ):
            for name in names:
                with self.subTest(step=name):
                    script = step_script(ROOT, name)
                    self.assertIn(f"$(python -I {SELECTOR} {runner})", script)


if __name__ == "__main__":
    unittest.main()
