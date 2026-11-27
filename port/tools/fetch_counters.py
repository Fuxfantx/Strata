"""Read a RenderDoc capture and print the GPU duration of every event.

This is the instrument the phase-4 goal was after: RenderDoc's replay API, driven from the x64 Python that runs under
Prism, over a capture taken from the x64 engine.  The counter is GPUCounter.EventGPUDuration, which the Adreno driver
reports through its own timestamp queries.

    python fetch_counters.py <capture.rdc> [top]
"""
import sys

sys.path.insert(0, r'C:\Users\Fuxfantx\A-Works\DSH Works\Strata\tools\rd\x64\Release\pymodules')
import renderdoc as rd


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    cap = rd.OpenCaptureFile()
    res = cap.OpenFile(path, '', None)
    if res != rd.ResultCode.Succeeded:
        print('OpenFile failed:', res)
        return 1
    status, controller = cap.OpenCapture(rd.ReplayOptions(), None)
    if status != rd.ResultCode.Succeeded:
        print('OpenCapture failed:', status)
        return 1

    print('capture:', path)
    print('driver :', controller.GetDriverInformation().driverName if hasattr(controller, 'GetDriverInformation') else '?')

    results = controller.FetchCounters([rd.GPUCounter.EventGPUDuration])
    rows = []
    for c in results:
        event = int(getattr(c, 'eventId', 0))
        # CounterValue is a tagged union: .d is the double, which for a duration counter is seconds
        raw = getattr(c, 'value', None)
        value = 0.0
        try:
            value = float(getattr(raw, 'd'))
        except Exception:
            try:
                value = float(raw)
            except Exception:
                value = 0.0
        rows.append((event, value * 1e6))   # seconds -> microseconds

    total = sum(us for _, us in rows)
    print(f'events with a duration: {len(rows)}, total {total / 1000.0:.3f} ms')
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f'{"us":>10}  {"event":>7}  name')
    for event, us in rows[:top]:
        name = ''
        try:
            action = controller.GetAction(event)
            if action is not None:
                name = action.GetName(controller.GetStructuredFile())
        except Exception:
            pass
        print(f'{us:10.3f}  {event:7d}  {name}')
    controller.Shutdown()
    cap.Shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
