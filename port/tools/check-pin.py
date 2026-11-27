"""Report how far the vendored llama.cpp pin is behind ggml-org/llama.cpp master.

The port keeps its own tree from a pin plus the overlay, so "syncing mainstream llama.cpp" is a re-pin, not a merge
(port/docs/MAINTAINER.md).  This tool answers the two questions that decide when to do it: what is pinned, and what has
landed since.  It only reads; it never changes the pin.

    python port/tools/check-pin.py            # latest master vs the pin
    python port/tools/check-pin.py --offline  # just print the pin

Exit code 0 when the pin is current, 1 when it is behind, 2 when the network lookup failed.
"""
import json
import pathlib
import re
import sys
import urllib.request

R = pathlib.Path(__file__).resolve().parents[2]
FETCH = R / 'port' / 'tools' / 'fetch-llama.ps1'
REPO = 'ggml-org/llama.cpp'


def pinned():
    text = FETCH.read_text(encoding='utf-8')
    m = re.search(r'\$Pin\s*=\s*"([0-9a-f]{40})"', text)
    if not m:
        sys.exit(f'cannot find $Pin in {FETCH}')
    repo = re.search(r'\$Repo\s*=\s*"([^"]+)"', text)
    return m.group(1), (repo.group(1) if repo else REPO)


def api(url):
    req = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json',
                                               'User-Agent': 'strata-vk-pin-check'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> int:
    pin, repo = pinned()
    print(f'pinned: {repo} @ {pin}')
    if '--offline' in sys.argv:
        return 0
    try:
        latest = api(f'https://api.github.com/repos/{repo}/commits?per_page=1')[0]
        head = latest['sha']
        when = latest['commit']['committer']['date']
        subject = latest['commit']['message'].splitlines()[0]
        if head == pin:
            print(f'up to date ({when})')
            return 0
        cmp = api(f'https://api.github.com/repos/{repo}/compare/{pin}...{head}')
        behind = cmp.get('total_commits', '?')
        print(f'master: {head} ({when}) - {behind} commit(s) ahead of the pin')
        print(f'  latest: {subject}')
        for c in cmp.get('commits', [])[-5:]:
            print(f'  {c["sha"][:10]} {c["commit"]["message"].splitlines()[0][:100]}')
        print('\nTo move: update $Pin in port/tools/fetch-llama.ps1, run it, resolve the overlay conflicts in',
              'port/llama-overlay-vanilla (never in third_party/llama.cpp), then run the gates in',
              'port/docs/MAINTAINER.md.')
        return 1
    except Exception as exc:  # noqa: BLE001 - any network/proxy failure is the same answer
        print(f'network lookup failed: {exc}')
        print('the pin is still valid; re-run when the network is back')
        return 2


if __name__ == '__main__':
    sys.exit(main())
