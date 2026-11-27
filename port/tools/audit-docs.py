"""Audit the port's documents: encoding, mojibake, PowerShell placeholders, broken links.

Scope is the port's own documentation: `port/docs/`, the two READMEs, `BUILD-WOA.md` and `UPDATES.md`. Application
documentation under `apps/` has its own conventions and is not checked here.

Checks:
  * decodes as UTF-8 with no BOM
  * no replacement characters or known mojibake sequences
  * no placeholder inside a PowerShell command (`<name>`, `<path>`, `<your-dir>`)
  * every relative markdown link resolves
"""
import os
import pathlib
import re
import sys

R = r"C:\Users\Fuxfantx\A-Works\DSH Works\Strata"

TARGETS = sorted(
    [str(q.relative_to(R)).replace(chr(92), '/') for q in pathlib.Path(R, 'port', 'docs').rglob('*.md')]
    + [f for f in ('port/README.md', 'README.md', 'README.zh-CN.md', 'BUILD-WOA.md', 'UPDATES.md')
       if pathlib.Path(R, f).is_file()]
)

MOJIBAKE = ["\ufffd", "\u00c3\u00a2", "\u00e2\u0080\u0099", "\u00f0\u009f", "\u00a4\u00a4"]

HTML_TAGS = {"picture", "source", "img", "sub", "sup", "br", "details", "summary", "p", "a",
             "h1", "h2", "h3", "h4", "h5", "h6", "b", "i", "code", "pre", "table", "tr", "td", "th"}
PLACEHOLDER = re.compile(r"<([A-Za-z][\w .-]{0,40})>")
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
HTML_REF = re.compile(r'(?:href|src)="([^"#]+)"')

problems = []
for rel in TARGETS:
    p = os.path.join(R, rel.replace("/", os.sep))
    if not os.path.exists(p):
        problems.append(f"{rel}: missing")
        continue
    raw = open(p, "rb").read()
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append(f"{rel}: UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        problems.append(f"{rel}: not valid UTF-8 ({e})")
        continue
    for m in MOJIBAKE:
        if m in text:
            line = text[:text.index(m)].count("\n") + 1
            problems.append(f"{rel}:L{line}: corruption sequence {m!r}")
    in_fence = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            for m in PLACEHOLDER.finditer(line):
                tag = m.group(1).lower()
                if tag in HTML_TAGS or tag.startswith("http"):
                    continue
                problems.append(f"{rel}:L{i}: placeholder <{m.group(1)}> in a command: {line.strip()[:60]}")
    for m in LINK.finditer(text):
        target = m.group(1)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        resolved = os.path.normpath(os.path.join(os.path.dirname(p), target.replace("/", os.sep)))
        if not os.path.exists(resolved):
            problems.append(f"{rel}: broken link {target}")
    for m in HTML_REF.finditer(text):
        target = m.group(1)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        resolved = os.path.normpath(os.path.join(os.path.dirname(p), target.replace("/", os.sep)))
        if not os.path.exists(resolved):
            problems.append(f"{rel}: broken reference {target}")

print(f"documents checked: {len(TARGETS)}")
if problems:
    print(f"problems: {len(problems)}")
    for x in problems:
        print("  ", x)
    sys.exit(1)
print("problems: 0")
