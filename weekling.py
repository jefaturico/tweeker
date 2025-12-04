from __future__ import annotations

import curses
import datetime as dt
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TODO_FILE = Path("todo.txt")


def week_start_for(date: dt.date) -> dt.date:
    return date - dt.timedelta(days=date.weekday())


@dataclass
class Task:
    date: dt.date
    text: str
    done: bool = False
    priority: str | None = None  # "A".."Z"

    def visible_body(self) -> str:
        tokens = [t for t in self.text.split() if not (t.startswith("+") or t.startswith("@"))]
        return " ".join(tokens)

    def display_text(self) -> str:
        prefix = "x " if self.done else ""
        prio = f"({self.priority}) " if self.priority else ""
        return f"{prefix}{prio}{self.visible_body()}".strip()

    def to_line(self) -> str:
        prefix = "x " if self.done else ""
        prio = f"({self.priority}) " if self.priority else ""
        return f"{prefix}{prio}{self.date.isoformat()} {self.text}".rstrip()

    def edit_text(self) -> str:
        prefix = "x " if self.done else ""
        prio = f"({self.priority}) " if self.priority else ""
        return f"{prefix}{prio}{self.text}".strip()


@dataclass
class State:
    tasks: Dict[dt.date, List[Task]] = field(default_factory=dict)
    week_start: dt.date = field(default_factory=lambda: week_start_for(dt.date.today()))
    cursor_day: int = 0
    cursor_idx: int = 0
    today: dt.date = field(default_factory=dt.date.today)
    search_query: str | None = None
    search_dir: int = 1

    def current_date(self) -> dt.date:
        return self.week_start + dt.timedelta(days=self.cursor_day)

    def current_tasks(self) -> List[Task]:
        return self.tasks.get(self.current_date(), [])

    def clamp_cursor(self) -> None:
        items = self.current_tasks()
        if not items:
            self.cursor_idx = 0
        else:
            self.cursor_idx = max(0, min(self.cursor_idx, len(items) - 1))


def load_tasks() -> Dict[dt.date, List[Task]]:
    tasks: Dict[dt.date, List[Task]] = {}
    if not TODO_FILE.exists():
        return tasks

    for raw in TODO_FILE.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        done = False
        priority = None
        if line.startswith("x "):
            done = True
            line = line[2:].lstrip()
        if line.startswith("(") and len(line) >= 3 and line[2] == ")":
            letter = line[1].upper()
            if "A" <= letter <= "Z":
                priority = letter
                line = line[3:].lstrip()
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            date = dt.date.fromisoformat(parts[0])
        except ValueError:
            continue
        text = parts[1] if len(parts) > 1 else ""
        tasks.setdefault(date, []).append(Task(date=date, text=text, done=done, priority=priority))
    return tasks


def save_tasks(tasks: Dict[dt.date, List[Task]]) -> None:
    lines: List[str] = []
    for date in sorted(tasks.keys()):
        for task in tasks.get(date, []):
            if not task.text.strip() and not task.done:
                continue
            lines.append(task.to_line())
    TODO_FILE.write_text("\n".join(lines) + ("\n" if lines else ""))


@dataclass
class KeyState:
    key: object = None
    time: float = 0.0


def read_key(win: curses.window, kstate: KeyState, *, debounce: float = 0.005, allow_repeat_keys=()) -> object:
    """Return a single key, ignoring identical repeats inside debounce window unless allowed."""
    while True:
        key = win.get_wch()
        now = time.monotonic()
        if (
            isinstance(key, str)
            and key == kstate.key
            and key not in allow_repeat_keys
            and (now - kstate.time) < debounce
        ):
            continue
        kstate.key = key
        kstate.time = now
        return key


def prompt_text(stdscr: curses.window, label: str, initial: str = "") -> str:
    """Inline prompt on the bottom row with the label tight to the left edge."""
    curses.curs_set(1)
    height, width = stdscr.getmaxyx()
    stdscr.move(height - 1, 0)
    stdscr.clrtoeol()

    prefix = f"{label}: "
    win = curses.newwin(1, width, height - 1, 0)
    win.keypad(True)

    text = list(initial)
    pos = len(text)
    kstate = KeyState()

    while True:
        win.erase()
        win.addnstr(0, 0, prefix, width - 1)
        field_start = len(prefix)
        visible = max(0, width - field_start - 1)
        offset = max(0, pos - visible)
        segment = "".join(text)[offset : offset + visible]
        win.addnstr(0, field_start, segment, width - field_start - 1)
        cursor_x = min(field_start + pos - offset, width - 1)
        win.move(0, cursor_x)

        try:
            key = read_key(win, kstate)
        except curses.error:
            continue

        if key in ("\n", "\r"):
            break
        if key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
            if pos > 0:
                text.pop(pos - 1)
                pos -= 1
            continue
        if key == curses.KEY_LEFT:
            pos = max(0, pos - 1)
            continue
        if key == curses.KEY_RIGHT:
            pos = min(len(text), pos + 1)
            continue
        if key == curses.KEY_HOME:
            pos = 0
            continue
        if key == curses.KEY_END:
            pos = len(text)
            continue
        if key == "\x1b":  # Esc cancels edits
            text = list(initial)
            break
        if isinstance(key, str) and key.isprintable():
            text.insert(pos, key)
            pos += 1

    curses.curs_set(0)
    return "".join(text).strip()


def parse_text_input(text: str) -> tuple[bool, str | None, str]:
    done = False
    priority = None
    body = text.strip()
    if body.startswith("x "):
        done = True
        body = body[2:].lstrip()
    if body.startswith("(") and len(body) >= 3 and body[2] == ")":
        letter = body[1].upper()
        if "A" <= letter <= "Z":
            priority = letter
            body = body[3:].lstrip()
    return done, priority, body


def parse_jump(text: str, today: dt.date) -> dt.date | None:
    text = text.strip()
    if not text:
        return None
    parts = text.split("-")
    try:
        if len(parts) == 3:
            return dt.date.fromisoformat(text)
        if len(parts) == 2:
            month = int(parts[0])
            day = int(parts[1])
            return dt.date(today.year, month, day)
        if len(parts) == 1 and text.isdigit():
            day = int(text)
            return dt.date(today.year, today.month, day)
    except ValueError:
        return None
    return None


def flatten_tasks(tasks: Dict[dt.date, List[Task]]) -> List[tuple[dt.date, int, Task]]:
    entries: List[tuple[dt.date, int, Task]] = []
    for date in sorted(tasks.keys()):
        for idx, task in enumerate(tasks[date]):
            entries.append((date, idx, task))
    return entries


def priority_value(letter: str | None) -> int:
    if letter is None:
        return 26
    return min(25, max(0, ord(letter.upper()) - ord("A")))


def sort_tasks_for_date(state: State, date: dt.date, focus_task: Task | None = None) -> None:
    items = state.tasks.get(date)
    if not items:
        return
    original_order = {id(task): i for i, task in enumerate(items)}
    items.sort(key=lambda t: (1 if t.done else 0, priority_value(t.priority), original_order.get(id(t), 0)))
    if focus_task:
        for i, t in enumerate(items):
            if t is focus_task:
                state.cursor_idx = i
                break
        else:
            state.clamp_cursor()
    else:
        state.clamp_cursor()


def find_match(state: State, query: str, direction: int, skip_current: bool = True) -> tuple[dt.date, int] | None:
    entries = flatten_tasks(state.tasks)
    if not entries:
        return None
    # find current position in flattened list
    current = state.current_date(), state.cursor_idx
    start_idx = 0
    for i, (d, idx, _) in enumerate(entries):
        if d == current[0] and idx == current[1]:
            start_idx = i
            break

    n = len(entries)
    q = query.lower()
    step = 1 if direction >= 0 else -1
    pos = start_idx
    first = True
    while True:
        if not (first and skip_current):
            _, _, task = entries[pos]
            if q in task.display_text().lower():
                return entries[pos][0], entries[pos][1]
        first = False
        pos = (pos + step) % n
        if pos == start_idx:
            break
    return None


def project_key(task: Task) -> str:
    for word in task.text.split():
        if word.startswith("+") and len(word) > 1:
            return word
    return None


def adjust_priority(task: Task, delta: int) -> None:
    val = priority_value(task.priority)
    new_val = val + delta
    if new_val > 26:
        new_val = 0
    elif new_val < 0:
        new_val = 26
    task.priority = None if new_val == 26 else chr(ord("A") + new_val)


def draw(stdscr: curses.window, state: State, status: str = "", show_help: bool = True) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    col_width = max(14, width // len(DAY_ORDER))
    base_y = 4

    for idx, day in enumerate(DAY_ORDER):
        x = idx * col_width
        date = state.week_start + dt.timedelta(days=idx)
        is_today = date == state.today
        is_active = idx == state.cursor_day
        header_attr = curses.A_BOLD | curses.color_pair(3)
        if is_today and is_active:
            header_attr = curses.A_BOLD | curses.color_pair(4)
        elif is_today:
            header_attr = curses.A_BOLD | curses.color_pair(4)
        elif is_active:
            header_attr = curses.A_BOLD | curses.color_pair(5)
        stdscr.addnstr(0, x, date.isoformat().center(col_width - 1), col_width - 1, header_attr)
        stdscr.addnstr(1, x, day.center(col_width - 1), col_width - 1, header_attr)

        tasks = state.tasks.get(date, [])
        no_project: list[tuple[Task, int]] = []
        seen_projects: list[str] = []
        grouped: list[tuple[str, list[tuple[Task, int]]]] = []
        for idx_t, t in enumerate(tasks):
            key = project_key(t)
            if key is None:
                no_project.append((t, idx_t))
                continue
            if key not in seen_projects:
                seen_projects.append(key)
                grouped.append((key, []))
            grouped[-1][1].append((t, idx_t))

        y = base_y

        def draw_block(label: str | None, items: list[tuple[Task, int]]) -> None:
            nonlocal y
            if not items:
                return
            if y >= height - 2:
                return
            label_text = (label or "").strip()
            if label_text:
                label_line = label_text.center(col_width - 1)
                stdscr.addnstr(y, x, label_line, col_width - 1, curses.color_pair(3) | curses.A_BOLD)
                y += 1
            for task, original_idx in items:
                if y >= height - 1:
                    break
                attr = curses.A_DIM
                if is_today:
                    attr = curses.color_pair(1)
                if is_active and original_idx == state.cursor_idx:
                    if is_today:
                        attr = curses.color_pair(4) | curses.A_BOLD | curses.A_STANDOUT
                    else:
                        attr = curses.color_pair(5) | curses.A_BOLD | curses.A_STANDOUT
                if task.done:
                    attr |= curses.A_DIM
                line = task.display_text()[: col_width - 1].ljust(col_width - 1)
                stdscr.addnstr(y, x, line, col_width - 1, attr)
                y += 1
            if y < height - 1:
                y += 1  # spacer between blocks

        # no-project items first, without label
        if no_project:
            draw_block("", no_project)
        for proj, items in grouped:
            label = proj[1:] if proj.startswith("+") else proj
            draw_block(label, items)


    base_help = "h/l day  j/k task  i edit  o/O new  dd delete  x done  +/- prio  w/b week  g goto  / ? search  n/N next  q quit"
    status_line = "" if not show_help and not status else (status or base_help)
    stdscr.move(height - 1, 0)
    stdscr.clrtoeol()
    week_text = f"W{state.week_start.isocalendar()[1]:02d}"
    usable_width = max(0, width - len(week_text) - 1)
    stdscr.addnstr(height - 1, 0, status_line, usable_width, curses.A_REVERSE)
    start_col = max(0, width - len(week_text) - 1)
    stdscr.addnstr(height - 1, start_col, week_text, len(week_text), curses.A_REVERSE)
    stdscr.refresh()


def insert_or_edit(stdscr: curses.window, state: State) -> None:
    items = state.tasks.setdefault(state.current_date(), [])
    current_text = items[state.cursor_idx].edit_text() if items else ""
    draw(stdscr, state, show_help=False)
    text = prompt_text(stdscr, "edit", current_text)
    if not text.strip():
        return
    done, priority, cleaned = parse_text_input(text)
    target_date = state.current_date()
    if items:
        task = items[state.cursor_idx]
        task.text = cleaned
        task.done = done
        task.priority = priority
        focus = task
    else:
        focus = Task(date=target_date, text=cleaned, done=done, priority=priority)
        items.append(focus)
        state.cursor_idx = 0
    sort_tasks_for_date(state, target_date, focus_task=focus)


def delete_current(state: State) -> None:
    items = state.current_tasks()
    if not items:
        return
    items.pop(state.cursor_idx)
    state.clamp_cursor()


def insert_at(stdscr: curses.window, state: State, idx: int) -> None:
    draw(stdscr, state, show_help=False)
    text = prompt_text(stdscr, "new")
    if not text.strip():
        return
    done, priority, cleaned = parse_text_input(text)
    items = state.tasks.setdefault(state.current_date(), [])
    task = Task(date=state.current_date(), text=cleaned, done=done, priority=priority)
    items.insert(idx, task)
    sort_tasks_for_date(state, state.current_date(), focus_task=task)


def main(stdscr: curses.window) -> None:
    curses.curs_set(0)
    curses.use_default_colors()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, -1)  # today tint (text)
    curses.init_pair(2, curses.COLOR_WHITE, -1)  # selection text
    curses.init_pair(3, curses.COLOR_WHITE, -1)  # headers/labels subtle
    curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_CYAN)  # today header bg
    curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE)  # active header bg (grey)
    stdscr.keypad(True)

    tasks = load_tasks()
    today = dt.date.today()
    today_idx = today.weekday()
    state = State(tasks=tasks, week_start=week_start_for(today), cursor_day=today_idx, today=today)
    for d in list(state.tasks.keys()):
        sort_tasks_for_date(state, d)

    pending_delete = False
    status = ""
    kstate = KeyState()
    while True:
        draw(stdscr, state, status)
        try:
            key = read_key(stdscr, kstate, allow_repeat_keys={"d"})
        except curses.error:
            continue

        if pending_delete:
            pending_delete = False
            if key == "d":
                delete_current(state)
                save_tasks(state.tasks)
                status = "deleted"
            else:
                status = ""
            continue

        status = ""
        if key in ("q", "\x1b"):
            break
        if key == "h":
            state.cursor_day = (state.cursor_day - 1) % len(DAY_ORDER)
            state.clamp_cursor()
        elif key == "l":
            state.cursor_day = (state.cursor_day + 1) % len(DAY_ORDER)
            state.clamp_cursor()
        elif key == "j":
            items = state.current_tasks()
            if items:
                state.cursor_idx = min(state.cursor_idx + 1, len(items) - 1)
        elif key == "k":
            if state.cursor_idx > 0:
                state.cursor_idx -= 1
        elif key == "i":
            insert_or_edit(stdscr, state)
            save_tasks(state.tasks)
        elif key == "o":
            items = state.tasks.setdefault(state.current_date(), [])
            insert_at(stdscr, state, state.cursor_idx + 1 if items else 0)
            save_tasks(state.tasks)
        elif key == "O":
            insert_at(stdscr, state, state.cursor_idx if state.current_tasks() else 0)
            save_tasks(state.tasks)
        elif key == "x":
            items = state.current_tasks()
            if items:
                task = items[state.cursor_idx]
                task.done = not task.done
                sort_tasks_for_date(state, state.current_date(), focus_task=task)
                save_tasks(state.tasks)
        elif key == "+":
            items = state.current_tasks()
            if items:
                task = items[state.cursor_idx]
                adjust_priority(task, -1)
                sort_tasks_for_date(state, state.current_date(), focus_task=task)
                save_tasks(state.tasks)
        elif key == "-":
            items = state.current_tasks()
            if items:
                task = items[state.cursor_idx]
                adjust_priority(task, 1)
                sort_tasks_for_date(state, state.current_date(), focus_task=task)
                save_tasks(state.tasks)
        elif key == "d":
            pending_delete = True
            status = "pending dd"
        elif key == "w":
            state.week_start = state.week_start + dt.timedelta(days=7)
            state.cursor_idx = 0
            state.clamp_cursor()
        elif key == "b":
            state.week_start = state.week_start - dt.timedelta(days=7)
            state.cursor_idx = 0
            state.clamp_cursor()
        elif key == "g":
            draw(stdscr, state, show_help=False)
            prompt_state = KeyState()
            try:
                first = read_key(stdscr, prompt_state, allow_repeat_keys={"g"})
            except curses.error:
                first = ""
            if first == "g":
                target = state.today
                state.week_start = week_start_for(target)
                state.cursor_day = target.weekday()
                state.clamp_cursor()
                status = "today"
            else:
                initial = first if isinstance(first, str) and first not in ("\n", "\r", "\x1b") else ""
                text = prompt_text(stdscr, "goto date", initial)
                target = parse_jump(text, state.today)
                if target:
                    state.week_start = week_start_for(target)
                    state.cursor_day = target.weekday()
                    state.clamp_cursor()
                    status = target.isoformat()
                else:
                    status = "invalid date"
        elif key == "/":
            draw(stdscr, state, show_help=False)
            q = prompt_text(stdscr, "search /")
            if q:
                state.search_query = q
                state.search_dir = 1
                found = find_match(state, q, 1)
                if found:
                    date, idx = found
                    state.week_start = week_start_for(date)
                    state.cursor_day = date.weekday()
                    state.cursor_idx = idx
                    status = f"{q}"
                else:
                    status = "no match"
        elif key == "?":
            draw(stdscr, state, show_help=False)
            q = prompt_text(stdscr, "search ?")
            if q:
                state.search_query = q
                state.search_dir = -1
                found = find_match(state, q, -1)
                if found:
                    date, idx = found
                    state.week_start = week_start_for(date)
                    state.cursor_day = date.weekday()
                    state.cursor_idx = idx
                    status = f"{q}"
                else:
                    status = "no match"
        elif key == "n":
            if not state.search_query:
                status = "no search"
            else:
                found = find_match(state, state.search_query, 1)
                if found:
                    date, idx = found
                    state.week_start = week_start_for(date)
                    state.cursor_day = date.weekday()
                    state.cursor_idx = idx
                    status = state.search_query
                else:
                    status = "no match"
        elif key == "N":
            if not state.search_query:
                status = "no search"
            else:
                found = find_match(state, state.search_query, -1)
                if found:
                    date, idx = found
                    state.week_start = week_start_for(date)
                    state.cursor_day = date.weekday()
                    state.cursor_idx = idx
                    status = state.search_query
                else:
                    status = "no match"


if __name__ == "__main__":
    curses.wrapper(main)
