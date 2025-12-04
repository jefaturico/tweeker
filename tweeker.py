from __future__ import annotations

import configparser
import curses
import datetime as dt
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

DEFAULT_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TODO_FILE = Path("todo.txt")
CONFIG_PATH = Path.home() / ".config" / "tweeker" / "config.ini"
DEFAULT_ARCHIVE_FILE = TODO_FILE.with_name("archive.txt")

COLOR_NAMES = {
    "black": curses.COLOR_BLACK,
    "red": curses.COLOR_RED,
    "green": curses.COLOR_GREEN,
    "yellow": curses.COLOR_YELLOW,
    "blue": curses.COLOR_BLUE,
    "magenta": curses.COLOR_MAGENTA,
    "cyan": curses.COLOR_CYAN,
    "white": curses.COLOR_WHITE,
    "default": -1,
    "none": -1,
}


def default_keybinds() -> Dict[str, list]:
    return {
        "quit": ["q"],
        "left": ["h", "KEY_LEFT"],
        "right": ["l", "KEY_RIGHT"],
        "down": ["j", "KEY_DOWN"],
        "up": ["k", "KEY_UP"],
        "select_range": ["v"],
        "select_single": ["V"],
        "undo": ["u"],
        "redo": ["U"],
        "edit": ["i"],
        "new_below": ["o"],
        "new_above": ["O"],
        "toggle_done": ["x"],
        "priority_up": ["+"],
        "priority_down": ["-"],
        "delete": ["d"],
        "copy": ["y"],
        "paste_after": ["p"],
        "paste_before": ["P"],
        "week_forward": ["w"],
        "week_back": ["b"],
        "goto_prefix": ["g"],
        "goto_date": ["G"],
        "jump_prefix": ["t"],
        "jump_date": ["T"],
        "search_forward": ["/"],
        "search_backward": ["?"],
        "search_next": ["n"],
        "search_prev": ["N"],
        "command": [":"],
        "details": ["\n", "\r", "KEY_ENTER", getattr(curses, "KEY_ENTER", 10), 10],
    }


def default_colors() -> Dict[str, str]:
    return {
        "default_fg": "white",
        "highlight_fg": "cyan",
    }


@dataclass
class Config:
    keybinds: Dict[str, set]
    todo_file: Path
    archive_file: Path
    default_fg: int
    highlight_fg: int
    unselected_dim: bool
    week_layout: List[str]
    first_weekday: int
    show_statusbar: bool = True
    move_overdue_to_today: bool = False
    archive_completed_after_days: int = 0
    save_debounce_seconds: float = 0.3
    expand_overflow_on_hover: bool = True
    days_on_screen: int = 7
    view_mode: str = "scroll"  # "scroll" or "week"


SPECIAL_KEYS = {
    "KEY_LEFT": curses.KEY_LEFT,
    "KEY_RIGHT": curses.KEY_RIGHT,
    "KEY_UP": curses.KEY_UP,
    "KEY_DOWN": curses.KEY_DOWN,
    "ESC": "\x1b",
    "ENTER": "\n",
    "KEY_ENTER": getattr(curses, "KEY_ENTER", 10),
}

DAY_NAME_TO_NUM = {name: idx for idx, name in enumerate(DEFAULT_DAY_ORDER)}


def load_parser_with_lines(path: Path) -> tuple[configparser.ConfigParser, dict[tuple[str, str], int], str | None]:
    """Read the INI file and capture line numbers for each option."""
    parser = configparser.ConfigParser()
    lines: dict[tuple[str, str], int] = {}
    if not path.exists():
        return parser, lines, None
    try:
        text = path.read_text()
    except Exception as exc:
        return parser, lines, f"could not read config: {exc}"
    try:
        parser.read_string(text)
    except Exception as exc:
        return parser, lines, f"could not parse config: {exc}"
    current_section = None
    for idx, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip().lower()
            continue
        if "=" in stripped or ":" in stripped:
            sep = "=" if "=" in stripped else ":"
            option = stripped.split(sep, 1)[0].strip().lower()
            if current_section:
                lines[(current_section, option)] = idx
    return parser, lines, None


def week_layout_for(start_on_sunday: bool) -> List[str]:
    if start_on_sunday:
        return ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    return list(DEFAULT_DAY_ORDER)


def day_names_for(start_date: dt.date, days: int) -> List[str]:
    names: List[str] = []
    for i in range(days):
        day = start_date + dt.timedelta(days=i)
        names.append(DEFAULT_DAY_ORDER[day.weekday()])
    return names


def effective_window_days(config: Config) -> int:
    return 7 if config.view_mode == "week" else max(1, config.days_on_screen)


def window_for_date(target: dt.date, config: Config) -> tuple[dt.date, int, int]:
    days = effective_window_days(config)
    if config.view_mode == "week":
        start = week_start_for(target, config.first_weekday)
        idx = weekday_index(target, config.first_weekday)
        return start, idx, days
    if days == 1:
        return target, 0, days
    if days == 2:
        return target, 0, days
    start = target - dt.timedelta(days=1)
    return start, 1, days


def parse_csv_list(csv: str | None) -> List[str]:
    if not csv:
        return []
    return [item.strip() for item in csv.split(",") if item.strip()]


MIN_COL_WIDTH = 12
MIN_HEIGHT = 27
MIN_WIDTH = 96


def resolve_color(value: str | int | None, fallback: str | int) -> int:
    candidate = fallback if value is None else value
    if isinstance(candidate, int):
        return candidate
    if isinstance(candidate, str):
        key = candidate.strip().lower()
        if key in COLOR_NAMES:
            return COLOR_NAMES[key]
        try:
            return int(key)
        except ValueError:
            pass
    return COLOR_NAMES["default"]


def normalize_key_token(token: object) -> object:
    if isinstance(token, int):
        return token
    if isinstance(token, str):
        trimmed = token.strip()
        if len(trimmed) == 1:
            return trimmed
        upper = trimmed.upper()
        if upper in SPECIAL_KEYS:
            return SPECIAL_KEYS[upper]
    return None


def normalize_keybinds(raw: Dict[str, list] | None) -> Dict[str, set]:
    merged: Dict[str, set] = {}
    for action, tokens in default_keybinds().items():
        merged[action] = set()
        incoming = tokens
        if raw and action in raw and isinstance(raw[action], list):
            incoming = raw[action]
        for tok in incoming:
            resolved = normalize_key_token(tok)
            if resolved is not None:
                merged[action].add(resolved)
    return merged


def default_config() -> Config:
    week_layout = week_layout_for(False)
    keybinds = normalize_keybinds(None)
    color_defaults = default_colors()
    default_fg = resolve_color(None, color_defaults["default_fg"])
    highlight_fg = resolve_color(None, color_defaults["highlight_fg"])
    return Config(
        keybinds=keybinds,
        todo_file=TODO_FILE,
        archive_file=DEFAULT_ARCHIVE_FILE,
        default_fg=default_fg,
        highlight_fg=highlight_fg,
        unselected_dim=True,
        week_layout=week_layout,
        first_weekday=DAY_NAME_TO_NUM.get(week_layout[0], 0),
        show_statusbar=True,
        move_overdue_to_today=False,
        archive_completed_after_days=0,
        save_debounce_seconds=0.3,
        expand_overflow_on_hover=True,
        days_on_screen=5,
        view_mode="scroll",
    )


def load_config() -> tuple[Config, list[str]]:
    parser, line_numbers, load_error = load_parser_with_lines(CONFIG_PATH)
    errors: list[str] = []
    if load_error:
        errors.append(load_error)

    start_on_sunday = False
    if parser.has_section("general"):
        try:
            start_on_sunday = parser.getboolean("general", "week_starts_on_sunday", fallback=False)
        except ValueError:
            ln = line_numbers.get(("general", "week_starts_on_sunday"))
            prefix = f"line {ln}: " if ln else ""
            errors.append(f"{prefix}week_starts_on_sunday must be true/false; using false")
            start_on_sunday = False
    week_layout = week_layout_for(start_on_sunday)

    raw_keybinds = None
    if parser.has_section("keybinds"):
        raw_keybinds = {}
        for action, tokens in parser.items("keybinds"):
            raw_keybinds[action] = parse_csv_list(tokens)
    keybinds = normalize_keybinds(raw_keybinds)
    conflicts = []
    owners: Dict[object, tuple[str, int | None]] = {}
    for action, keys in keybinds.items():
        ln = line_numbers.get(("keybinds", action))
        for key in keys:
            previous = owners.get(key)
            if previous and previous[0] != action:
                prev_action, prev_line = previous
                conflicts.append(f"line {ln or '?'}: {format_key(key)} also assigned to {prev_action} (line {prev_line or '?'})")
            owners[key] = (action, ln)
    if conflicts:
        errors.extend(conflicts)

    color_defaults = default_colors()
    def parse_color(option: str, fallback_key: str) -> int:
        raw = parser.get("colors", option, fallback=None) if parser.has_section("colors") else None
        fallback = color_defaults[fallback_key]
        resolved = resolve_color(raw, fallback)
        if raw is not None:
            candidate = raw.strip().lower()
            if candidate not in COLOR_NAMES:
                try:
                    int(candidate)
                except ValueError:
                    ln = line_numbers.get(("colors", option))
                    prefix = f"line {ln}: " if ln else ""
                    errors.append(f"{prefix}colors.{option} '{raw}' is invalid; using {fallback}")
        return resolved

    default_fg = parse_color("default_fg", "default_fg")
    highlight_fg = parse_color("highlight_fg", "highlight_fg")
    unselected_dim = True

    todo_file = TODO_FILE
    if parser.has_section("general"):
        raw_path = parser.get("general", "todo_file", fallback=None)
        if raw_path:
            ln = line_numbers.get(("general", "todo_file"))
            try:
                candidate = Path(raw_path.strip()).expanduser()
                if candidate.is_dir():
                    prefix = f"line {ln}: " if ln else ""
                    errors.append(f"{prefix}todo_file points to a directory; using {TODO_FILE}")
                else:
                    todo_file = candidate
            except Exception as exc:
                prefix = f"line {ln}: " if ln else ""
                errors.append(f"{prefix}todo_file invalid ({exc}); using {TODO_FILE}")
    archive_file = todo_file.with_name("archive.txt")
    if parser.has_section("general"):
        raw_path = parser.get("general", "archive_file", fallback=None)
        if raw_path:
            ln = line_numbers.get(("general", "archive_file"))
            try:
                candidate = Path(raw_path.strip()).expanduser()
                if candidate.is_dir():
                    prefix = f"line {ln}: " if ln else ""
                    errors.append(f"{prefix}archive_file points to a directory; using {DEFAULT_ARCHIVE_FILE}")
                else:
                    archive_file = candidate
            except Exception as exc:
                prefix = f"line {ln}: " if ln else ""
                errors.append(f"{prefix}archive_file invalid ({exc}); using {DEFAULT_ARCHIVE_FILE}")

    archive_after_days = 0
    if parser.has_section("general"):
        raw_days = parser.get("general", "archive_completed_after_days", fallback=None)
        if raw_days:
            ln = line_numbers.get(("general", "archive_completed_after_days"))
            try:
                archive_after_days = max(0, int(raw_days))
            except ValueError:
                prefix = f"line {ln}: " if ln else ""
                errors.append(f"{prefix}archive_completed_after_days must be an integer; using 0")
                archive_after_days = 0

    save_debounce_seconds = 0.3
    if parser.has_section("general"):
        raw = parser.get("general", "save_debounce_seconds", fallback=None)
        if raw:
            ln = line_numbers.get(("general", "save_debounce_seconds"))
            try:
                save_debounce_seconds = max(0.0, float(raw))
            except ValueError:
                prefix = f"line {ln}: " if ln else ""
                errors.append(f"{prefix}save_debounce_seconds must be a number; using 0.3")
                save_debounce_seconds = 0.3

    expand_overflow_on_hover = True
    expand_ln = line_numbers.get(("general", "expand_overflow_on_hover"))
    expand_raw = parser.get("general", "expand_overflow_on_hover", fallback="true") if parser.has_section("general") else "true"
    try:
        expand_overflow_on_hover = parser.getboolean("general", "expand_overflow_on_hover", fallback=True)
    except ValueError:
        prefix = f"line {expand_ln}: " if expand_ln else ""
        errors.append(f"{prefix}expand_overflow_on_hover must be true/false; using true")
        expand_overflow_on_hover = True

    days_on_screen = 7
    if parser.has_section("general"):
        raw_days = parser.get("general", "days_on_screen", fallback=None)
        if raw_days:
            ln = line_numbers.get(("general", "days_on_screen"))
            try:
                days_on_screen = max(1, int(raw_days))
            except ValueError:
                prefix = f"line {ln}: " if ln else ""
                errors.append(f"{prefix}days_on_screen must be an integer; using 7")
                days_on_screen = 7

    view_mode = "scroll"
    if parser.has_section("general"):
        raw_mode = parser.get("general", "view_mode", fallback="scroll").strip().lower()
        valid_modes = {"scroll", "week"}
        if raw_mode in valid_modes:
            view_mode = raw_mode
        else:
            ln = line_numbers.get(("general", "view_mode"))
            prefix = f"line {ln}: " if ln else ""
            errors.append(f"{prefix}view_mode must be 'scroll' or 'week'; using scroll")
            view_mode = "scroll"

    def parse_bool(section: str, option: str, fallback: bool) -> bool:
        if not parser.has_section(section):
            return fallback
        try:
            return parser.getboolean(section, option, fallback=fallback)
        except ValueError:
            ln = line_numbers.get((section, option))
            prefix = f"line {ln}: " if ln else ""
            errors.append(f"{prefix}{section}.{option} must be true/false; using {fallback}")
            return fallback

    first_weekday = DAY_NAME_TO_NUM.get(week_layout[0], 0)
    show_statusbar = parse_bool("general", "show_statusbar", True)
    move_overdue_to_today = parse_bool("general", "move_overdue_to_today", False)

    if errors:
        return default_config(), errors

    return Config(
        keybinds=keybinds,
        todo_file=todo_file,
        archive_file=archive_file,
        default_fg=default_fg,
        highlight_fg=highlight_fg,
        unselected_dim=unselected_dim,
        week_layout=week_layout,
        first_weekday=first_weekday,
        show_statusbar=show_statusbar,
        move_overdue_to_today=move_overdue_to_today,
        archive_completed_after_days=archive_after_days,
        save_debounce_seconds=save_debounce_seconds,
        expand_overflow_on_hover=expand_overflow_on_hover,
        days_on_screen=days_on_screen,
        view_mode=view_mode,
    ), []


def week_start_for(date: dt.date, first_weekday: int = 0) -> dt.date:
    delta = (date.weekday() - first_weekday) % 7
    return date - dt.timedelta(days=delta)


def weekday_index(date: dt.date, first_weekday: int = 0) -> int:
    return (date.weekday() - first_weekday) % 7


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
class Snapshot:
    tasks: Dict[dt.date, List[Task]]
    week_start: dt.date
    cursor_day: int
    cursor_idx: int


@dataclass
class State:
    tasks: Dict[dt.date, List[Task]] = field(default_factory=dict)
    week_start: dt.date = field(default_factory=lambda: week_start_for(dt.date.today()))
    cursor_day: int = 0
    cursor_idx: int = 0
    today: dt.date = field(default_factory=dt.date.today)
    search_query: str | None = None
    search_dir: int = 1
    day_order: List[str] = field(default_factory=lambda: list(DEFAULT_DAY_ORDER))
    first_weekday: int = 0
    yank: List[Task] | None = None
    config: Config | None = None
    selected_ids: set[int] = field(default_factory=set)
    selection_mode: str = "none"  # "none", "range", "multi"
    selection_anchor: int | None = None  # task id for range anchor
    undo_stack: List[Snapshot] = field(default_factory=list)
    redo_stack: List[Snapshot] = field(default_factory=list)
    dirty: bool = False
    last_save_time: float = 0.0

    def update_week_window(self, new_start: dt.date | None = None, cursor_day: int | None = None) -> None:
        if new_start:
            self.week_start = new_start
        if not self.config:
            return
        days = effective_window_days(self.config)
        start = self.week_start
        if self.config.view_mode == "week":
            start = week_start_for(start, self.first_weekday)
        self.week_start = start
        self.day_order = day_names_for(self.week_start, days)
        max_idx = max(0, len(self.day_order) - 1)
        if cursor_day is not None:
            self.cursor_day = max(0, min(cursor_day, max_idx))
        else:
            self.cursor_day = max(0, min(self.cursor_day, max_idx))
        self.clamp_cursor()

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


def load_tasks(todo_path: Path) -> Dict[dt.date, List[Task]]:
    """Load tasks from the todo file path; tolerates missing files."""
    tasks: Dict[dt.date, List[Task]] = {}
    if not todo_path.exists():
        return tasks

    for raw in todo_path.read_text().splitlines():
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


def save_tasks(tasks: Dict[dt.date, List[Task]], todo_path: Path) -> None:
    """Persist tasks to the todo file path, creating parent directories as needed."""
    lines: List[str] = []
    for date in sorted(tasks.keys()):
        for task in tasks.get(date, []):
            if not task.text.strip() and not task.done:
                continue
            lines.append(task.to_line())
    todo_path.parent.mkdir(parents=True, exist_ok=True)
    todo_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def mark_dirty(state: State) -> None:
    state.dirty = True


def persist_tasks_if_needed(state: State, force: bool = False) -> None:
    if not state.config:
        return
    now = time.monotonic()
    should_save = force or (state.dirty and (now - state.last_save_time) >= state.config.save_debounce_seconds)
    if not should_save:
        return
    if state.config.archive_completed_after_days > 0:
        cutoff = dt.date.today() - dt.timedelta(days=state.config.archive_completed_after_days)
        moved = archive_completed(state.tasks, state.config.archive_file, cutoff)
        if moved:
            state.dirty = True
    save_tasks(state.tasks, state.config.todo_file)
    state.last_save_time = time.monotonic()
    state.dirty = False


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

    if label.endswith(":"):
        prefix = f"{label} "
    else:
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
        if key == "\x1b":  # Esc cancels edits immediately
            text = []
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
    def sort_key(t: Task) -> tuple:
        ctx = context_key(t)
        ctx_label = ctx[1:] if ctx and ctx.startswith("@") else (ctx or "")
        return (
            ctx_label.lower(),
            1 if t.done else 0,
            priority_value(t.priority),
            original_order.get(id(t), 0),
        )

    items.sort(key=sort_key)
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
            haystack = f"{task.display_text()} {task.text}".lower()
            if q in haystack:
                return entries[pos][0], entries[pos][1]
        first = False
        pos = (pos + step) % n
        if pos == start_idx:
            break
    return None


def context_key(task: Task) -> str:
    for word in task.text.split():
        if word.startswith("@") and len(word) > 1:
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
    required_width = max(MIN_WIDTH, MIN_COL_WIDTH * len(state.day_order))
    required_height = MIN_HEIGHT
    if width < required_width or height < required_height:
        msg = f"tweeker needs at least {required_width}x{required_height}. current: {width}x{height}"
        hint = "resize your terminal to continue"
        y = height // 2 - 1
        stdscr.addnstr(max(0, y), 1, msg[: max(0, width - 2)], max(0, width - 2), curses.A_BOLD)
        stdscr.addnstr(max(0, y + 1), 1, hint[: max(0, width - 2)], max(0, width - 2))
        stdscr.refresh()
        return
    border_attr = curses.color_pair(1) | curses.A_DIM
    try:
        stdscr.hline(0, 0, curses.ACS_HLINE | border_attr, width)
        stdscr.hline(height - 1, 0, curses.ACS_HLINE | border_attr, width)
        stdscr.vline(0, 0, curses.ACS_VLINE | border_attr, height)
        stdscr.vline(0, width - 1, curses.ACS_VLINE | border_attr, height)
        stdscr.addch(0, 0, curses.ACS_ULCORNER | border_attr)
        stdscr.addch(0, width - 1, curses.ACS_URCORNER | border_attr)
        stdscr.addch(height - 1, 0, curses.ACS_LLCORNER | border_attr)
        stdscr.addch(height - 1, width - 1, curses.ACS_LRCORNER | border_attr)
    except curses.error:
        pass

    inner_width = max(1, width - 2)
    col_width = max(MIN_COL_WIDTH, inner_width // len(state.day_order))
    base_y = 5
    top_offset = 1

    status_y = height - 2
    # vertical separators between days (stop before status bar and bottom line)
    for idx in range(1, len(state.day_order)):
        sep_x = 1 + idx * col_width
        if sep_x < 0 or sep_x >= width:
            continue
        sep_attr = curses.color_pair(1) | curses.A_DIM
        for y in range(1, status_y):
            try:
                stdscr.addch(y, sep_x, curses.ACS_VLINE | sep_attr)
            except curses.error:
                continue
    # horizontal separator between headers and tasks
    sep_y = 3
    try:
        stdscr.hline(sep_y, 1, curses.ACS_HLINE | (curses.color_pair(1) | curses.A_DIM), max(0, inner_width))
    except curses.error:
        pass
    base_y = sep_y + 1
    # horizontal separator above status bar
    try:
        stdscr.hline(status_y - 1, 1, curses.ACS_HLINE | (curses.color_pair(1) | curses.A_DIM), max(0, inner_width))
    except curses.error:
        pass

    for idx, day in enumerate(state.day_order):
        x = 1 + idx * col_width
        gutter_left = 2
        gutter_right = 1
        content_w = max(1, col_width - gutter_left - gutter_right)
        start_x = x + gutter_left
        date = state.week_start + dt.timedelta(days=idx)
        is_today = date == state.today
        is_active = idx == state.cursor_day
        base_attr = curses.color_pair(2 if is_today else 1)
        if not is_active and state.config and state.config.unselected_dim:
            base_attr |= curses.A_DIM
        header_attr = base_attr | curses.A_BOLD
        stdscr.addnstr(top_offset, start_x, date.isoformat().rjust(content_w), content_w, header_attr)
        stdscr.addnstr(top_offset + 1, start_x, day.ljust(content_w), content_w, header_attr)

        tasks = state.tasks.get(date, [])
        no_project: list[tuple[Task, int]] = []
        seen_projects: list[str] = []
        grouped: list[tuple[str, list[tuple[Task, int]]]] = []
        for idx_t, t in enumerate(tasks):
            key = context_key(t)
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
                label_line = label_text.center(content_w)
                stdscr.addnstr(y, start_x, label_line, content_w, base_attr | curses.A_BOLD)
                y += 1
            for task, original_idx in items:
                if y >= height - 1:
                    break
                attr = base_attr
                is_overdue = not task.done and task.date < state.today
                if is_overdue:
                    attr = curses.color_pair(4)
                if id(task) in state.selected_ids:
                    attr |= curses.A_REVERSE | curses.A_BOLD
                if is_active and original_idx == state.cursor_idx:
                    if is_overdue:
                        attr = curses.color_pair(4) | curses.A_BOLD | curses.A_STANDOUT
                    else:
                        active_attr = curses.color_pair(2 if is_today else 1) | curses.A_BOLD | curses.A_STANDOUT
                        attr = active_attr
                if task.done:
                    attr |= curses.A_DIM
                text = task.display_text()
                is_hovered = is_active and original_idx == state.cursor_idx
                pad_left = 1 if is_hovered else 0
                usable_text_width = max(1, content_w - pad_left)
                expanded = state.config.expand_overflow_on_hover and is_hovered and len(text) > usable_text_width
                lines = textwrap.wrap(text, width=usable_text_width) if expanded else [text[:usable_text_width]]
                for idx_line, line in enumerate(lines):
                    if y >= height - 1:
                        break
                    prefix = " " if is_hovered else ""
                    padded = f"{prefix}{line}".ljust(content_w)
                    stdscr.addnstr(y, start_x, padded, content_w, attr)
                    y += 1
            if y < height - 1:
                y += 1  # spacer between blocks

        # no-project items first, without label
        if no_project:
            draw_block("", no_project)
        for proj, items in grouped:
            label = proj[1:] if proj.startswith("@") else proj
            draw_block(label, items)


    base_help = "q quit  h/l day  j/k task  o/O new  dd delete  yy copy  v/V select  u/U undo/redo  g goto  :help"
    status_line = "" if not show_help and not status else (status or base_help)
    if state.config and not state.config.show_statusbar and not status:
        stdscr.refresh()
        return
    # clear status interior without touching borders
    inner_width = max(0, width - 2)
    stdscr.addnstr(status_y, 1, " " * inner_width, inner_width)
    week_text = f"Week {state.week_start.isocalendar()[1]:02d}"
    usable_width = max(0, inner_width - len(week_text) - 1)
    status_x = 2  # slight padding from left border
    status_attr = (curses.color_pair(1) | curses.A_DIM)
    if state.config and state.config.show_statusbar:
        stdscr.addnstr(status_y, status_x, status_line, usable_width, status_attr)
    elif status:
        stdscr.addnstr(status_y, status_x, status_line, usable_width, status_attr)
    start_col = max(1, width - len(week_text) - 2)
    if start_col < width - 1:
        stdscr.addnstr(status_y, start_col, week_text, min(len(week_text), width - 1 - start_col), status_attr | curses.A_BOLD)
    # bottom line below status
    try:
        stdscr.hline(height - 1, 1, curses.ACS_HLINE | (curses.color_pair(1) | curses.A_DIM), max(0, width - 2))
        stdscr.addch(height - 1, 0, curses.ACS_LLCORNER | border_attr)
        stdscr.addch(height - 1, width - 1, curses.ACS_LRCORNER | border_attr)
    except curses.error:
        pass
    stdscr.refresh()


def insert_or_edit(stdscr: curses.window, state: State) -> None:
    items = state.tasks.setdefault(state.current_date(), [])
    current_text = items[state.cursor_idx].edit_text() if items else ""
    draw(stdscr, state, show_help=False)
    text = prompt_text(stdscr, "edit", current_text)
    if not text.strip():
        return
    done, priority, cleaned = parse_text_input(text)
    record_undo(state)
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


def edit_tasks(stdscr: curses.window, state: State, targets: List[Task]) -> None:
    if not targets:
        return
    if len(targets) == 1:
        initial = targets[0].edit_text()
        draw(stdscr, state, show_help=False)
        text = prompt_text(stdscr, "edit", initial)
        if not text.strip():
            return
        record_undo(state)
        done, priority, cleaned = parse_text_input(text)
        task = targets[0]
        task.text = cleaned
        task.done = done
        task.priority = priority
        sort_tasks_for_date(state, task.date, focus_task=task)
        return

    def split_body_and_tags(text: str) -> tuple[str, List[str]]:
        tokens = text.split()
        body = [t for t in tokens if not (t.startswith("@") or t.startswith("+"))]
        tags = [t for t in tokens if t.startswith("@") or t.startswith("+")]
        return " ".join(body).strip(), tags

    base_body, base_tags = split_body_and_tags(targets[0].text)
    draw(stdscr, state, show_help=False)
    text = prompt_text(stdscr, "edit tags", " ".join(base_tags))
    if text is None:
        return
    new_tags = [t for t in text.split() if t.startswith("@") or t.startswith("+")]
    record_undo(state)
    for task in targets:
        body, _ = split_body_and_tags(task.text)
        tag_part = " ".join(new_tags).strip()
        combined = body
        if tag_part:
            combined = f"{body} {tag_part}".strip()
        task.text = combined
        sort_tasks_for_date(state, task.date, focus_task=task)


def delete_tasks(state: State, tasks_to_delete: List[Task]) -> None:
    if not tasks_to_delete:
        return
    target_ids = {id(t) for t in tasks_to_delete}
    for date, items in list(state.tasks.items()):
        state.tasks[date] = [t for t in items if id(t) not in target_ids]
        if not state.tasks[date]:
            state.tasks.pop(date, None)
    state.clamp_cursor()


def yank_current(state: State) -> None:
    entries = selected_entries(state)
    if not entries:
        cur = current_task(state)
        if cur:
            entries = [(state.current_date(), cur)]
    if not entries:
        return
    state.yank = []
    for date, task in entries:
        state.yank.append(Task(date=date, text=task.text, done=task.done, priority=task.priority))


def paste_task(state: State, *, before: bool) -> Task | None:
    if not state.yank:
        return None
    record_undo(state)
    items = state.tasks.setdefault(state.current_date(), [])
    insert_at = state.cursor_idx if before else state.cursor_idx + 1
    insert_at = max(0, min(insert_at, len(items)))
    last_inserted = None
    for clone_src in state.yank:
        clone = Task(
            date=state.current_date(),
            text=clone_src.text,
            done=clone_src.done,
            priority=clone_src.priority,
        )
        items.insert(insert_at, clone)
        insert_at += 1
        last_inserted = clone
    sort_tasks_for_date(state, state.current_date(), focus_task=last_inserted)
    return last_inserted


def contexts_for_task(task: Task) -> List[str]:
    return [w for w in task.text.split() if w.startswith("@") and len(w) > 1]


def projects_for_task(task: Task) -> List[str]:
    return [w for w in task.text.split() if w.startswith("+") and len(w) > 1]


def tasks_for_context(state: State, context: str) -> List[Task]:
    hits: list[Task] = []
    for tasks in state.tasks.values():
        for t in tasks:
            if context in contexts_for_task(t):
                hits.append(t)
    return hits


def tasks_for_project(state: State, project: str) -> List[Task]:
    hits: list[Task] = []
    for tasks in state.tasks.values():
        for t in tasks:
            if project in projects_for_task(t):
                hits.append(t)
    return hits


def handle_overdue_tasks(tasks: Dict[dt.date, List[Task]], today: dt.date, move_to_today: bool) -> bool:
    if not tasks:
        return False
    today_list = tasks.setdefault(today, [])
    changed = False
    for date in list(tasks.keys()):
        items = tasks.get(date, [])
        i = 0
        while i < len(items):
            task = items[i]
            if not task.done and task.date < today:
                if move_to_today:
                    task.date = today
                    today_list.append(task)
                    items.pop(i)
                    changed = True
                    continue
            i += 1
        if not items:
            tasks.pop(date, None)
            changed = True
    return changed


def archive_completed(tasks: Dict[dt.date, List[Task]], archive_path: Path, cutoff: dt.date) -> bool:
    """Move completed tasks older than cutoff to archive file. Returns True if any moved."""
    moved: list[str] = []
    for date in list(tasks.keys()):
        items = tasks.get(date, [])
        keep = []
        for task in items:
            if task.done and task.date < cutoff:
                moved.append(task.to_line())
            else:
                keep.append(task)
        if keep:
            tasks[date] = keep
        else:
            tasks.pop(date, None)
    if not moved:
        return False
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("a", encoding="utf-8") as f:
        for line in moved:
            f.write(line.rstrip("\n") + "\n")
    return True


def move_tasks_to_date(state: State, target_date: dt.date, tasks_to_move: List[Task]) -> bool:
    if not tasks_to_move:
        return False
    record_undo(state)
    target_ids = {id(t) for t in tasks_to_move}
    for date, items in list(state.tasks.items()):
        state.tasks[date] = [t for t in items if id(t) not in target_ids]
        if not state.tasks[date]:
            state.tasks.pop(date, None)
    dest_list = state.tasks.setdefault(target_date, [])
    for t in tasks_to_move:
        t.date = target_date
        dest_list.append(t)
    sort_tasks_for_date(state, target_date)
    new_start, new_idx, _ = window_for_date(target_date, state.config)
    state.update_week_window(new_start, cursor_day=new_idx)
    state.cursor_idx = 0
    state.clamp_cursor()
    mark_dirty(state)
    return True


def clone_tasks(tasks: Dict[dt.date, List[Task]]) -> Dict[dt.date, List[Task]]:
    out: Dict[dt.date, List[Task]] = {}
    for date, items in tasks.items():
        out[date] = [Task(date=t.date, text=t.text, done=t.done, priority=t.priority) for t in items]
    return out


def snapshot_state(state: State) -> Snapshot:
    return Snapshot(
        tasks=clone_tasks(state.tasks),
        week_start=state.week_start,
        cursor_day=state.cursor_day,
        cursor_idx=state.cursor_idx,
    )


def apply_snapshot(state: State, snap: Snapshot) -> None:
    state.tasks = snap.tasks
    state.week_start = snap.week_start
    state.cursor_day = snap.cursor_day
    state.cursor_idx = snap.cursor_idx
    state.selected_ids.clear()
    state.selection_mode = "none"
    state.selection_anchor = None
    state.update_week_window(state.week_start, state.cursor_day)
    state.clamp_cursor()


def record_undo(state: State) -> None:
    state.undo_stack.append(snapshot_state(state))
    state.redo_stack.clear()


def current_task(state: State) -> Task | None:
    items = state.current_tasks()
    if not items:
        return None
    if state.cursor_idx < 0 or state.cursor_idx >= len(items):
        return None
    return items[state.cursor_idx]


def flatten_with_ids(state: State) -> List[tuple[int, dt.date, int, Task]]:
    entries: list[tuple[int, dt.date, int, Task]] = []
    for date in sorted(state.tasks.keys()):
        for idx, task in enumerate(state.tasks[date]):
            entries.append((id(task), date, idx, task))
    return entries


def selected_entries(state: State) -> List[tuple[dt.date, Task]]:
    out: list[tuple[dt.date, Task]] = []
    if not state.selected_ids:
        return out
    for date, tasks in state.tasks.items():
        for t in tasks:
            if id(t) in state.selected_ids:
                out.append((date, t))
    return out


def clear_selection(state: State) -> None:
    state.selected_ids.clear()
    state.selection_mode = "none"
    state.selection_anchor = None


def update_range_selection(state: State) -> None:
    if state.selection_mode != "range" or state.selection_anchor is None:
        return
    cur = current_task(state)
    if not cur:
        state.selected_ids.clear()
        return
    entries = flatten_with_ids(state)
    pos = {tid: i for i, (tid, _, _, _) in enumerate(entries)}
    anchor_idx = pos.get(state.selection_anchor)
    cur_idx = pos.get(id(cur))
    if anchor_idx is None or cur_idx is None:
        return
    lo, hi = sorted((anchor_idx, cur_idx))
    state.selected_ids = {entries[i][0] for i in range(lo, hi + 1)}


def format_key(token: object) -> str:
    reverse_special = {v: k.lower() for k, v in SPECIAL_KEYS.items()}
    if isinstance(token, str):
        if token == "\n":
            return "enter"
        if token == "\x1b":
            return "esc"
        return token
    if isinstance(token, int):
        if token in reverse_special:
            return reverse_special[token]
        if token == 10:
            return "enter"
        return f"key-{token}"
    return "?"


def show_keybinds(stdscr: curses.window, config: Config) -> None:
    height, width = stdscr.getmaxyx()
    win = curses.newwin(height, width, 0, 0)
    win.erase()
    win.border()
    title = " keybinds (press any key to close) "
    win.addnstr(0, max(1, (width - len(title)) // 2), title, len(title), curses.A_BOLD)
    friendly_names = {
        "quit": "quit",
        "left": "move day left",
        "right": "move day right",
        "down": "next task",
        "up": "prev task",
        "edit": "edit task",
        "new_below": "new task below",
        "new_above": "new task above",
        "toggle_done": "toggle done",
        "priority_up": "priority up",
        "priority_down": "priority down",
        "delete": "delete (dd)",
        "copy": "copy (yy)",
        "paste_after": "paste after",
        "paste_before": "paste before",
        "week_forward": "next week",
        "week_back": "prev week",
        "goto_prefix": "goto (days)",
        "goto_date": "goto date (G)",
        "jump_prefix": "jump (t)",
        "jump_date": "jump to date (T)",
        "search_forward": "search forward",
        "search_backward": "search backward",
        "search_next": "next match",
        "search_prev": "prev match",
        "command": "command (:)",
        "select_range": "select range (v)",
        "select_single": "select toggle (V)",
        "undo": "undo",
        "redo": "redo",
    }
    actions = list(config.keybinds.items())
    actions.sort(key=lambda pair: pair[0])
    row = 1
    for action, keys in actions:
        if row >= height - 2:
            break
        label = friendly_names.get(action, action).ljust(18)
        key_text = ", ".join(format_key(k) for k in sorted(keys, key=str))
        win.addnstr(row, 2, f"{label} {key_text}", width - 4)
        row += 1
    win.refresh()
    kstate = KeyState()
    try:
        read_key(win, kstate)
    except curses.error:
        pass


def modal_geometry(stdscr: curses.window) -> tuple[int, int, int, int]:
    height, width = stdscr.getmaxyx()
    win_h = min(height - 8, max(14, height * 2 // 3))
    win_w = min(width - 8, max(40, int(width * 0.55)))
    start_y = max(3, (height - win_h) // 2)
    start_x = max(3, (width - win_w) // 2)
    return win_h, win_w, start_y, start_x


def compact_modal_geometry(stdscr: curses.window) -> tuple[int, int, int, int]:
    height, width = stdscr.getmaxyx()
    win_h = min(height - 6, max(8, height // 3))
    win_w = min(width - 6, max(50, int(width * 0.7)))
    start_y = max(2, (height - win_h) // 2)
    start_x = max(2, (width - win_w) // 2)
    return win_h, win_w, start_y, start_x


def show_task_details(stdscr: curses.window, state: State) -> None:
    items = state.current_tasks()
    if not items:
        return
    task = items[state.cursor_idx]
    height, width = stdscr.getmaxyx()
    body = task.visible_body().strip() or "(empty)"
    wrapped = textwrap.wrap(body, width=max(20, (width - 6) // 2))
    projects = projects_for_task(task)
    content_rows = 4 + len(wrapped)  # due + blank + "Task:" + wrapped + trailing blank
    if projects:
        project = projects[0]
        related = tasks_for_project(state, project)
        completed = [t for t in related if t.done]
        pending = [t for t in related if not t.done]
        content_rows += len(pending) + len(completed) + 6  # header + blank + labels + spacer per section

    win_h = min(max(6, content_rows + 2), height - 2)
    win_w = min(width - 4, max(40, width // 2))
    start_y = max(1, (height - win_h) // 2)
    start_x = max(2, (width - win_w) // 2)
    win = curses.newwin(win_h, win_w, start_y, start_x)
    win.erase()
    win.border()
    title = " Task Details (press any key to close) "
    win.addnstr(0, max(1, (win_w - len(title)) // 2), title, len(title), curses.A_BOLD)

    lines: list[str] = []
    lines.append(f"Due: {task.date.isoformat()} ({task.date.strftime('%A')})")
    lines.append("")
    if wrapped:
        lines.append(f"Task: {wrapped[0]}")
        lines.extend([f" {w}" for w in wrapped[1:]])
    else:
        lines.append("Task:")
    lines.append("")

    projects = projects_for_task(task)
    row = 1
    for entry in lines:
        if row >= win_h - 2:
            break
        if entry.startswith("Task: "):
            label = "Task:"
            rest = entry[len("Task: ") :]
            win.addnstr(row, 2, label, win_w - 4, curses.A_BOLD)
            if rest and len(label) + 1 < win_w - 4:
                win.addnstr(row, 2 + len(label) + 1, rest[: win_w - 4 - len(label) - 1], win_w - 4 - len(label) - 1)
        else:
            attr = curses.A_BOLD if entry and not entry.startswith("  ") else 0
            win.addnstr(row, 2, entry[: win_w - 4], win_w - 4, attr)
        row += 1

    if projects and row < win_h - 3:
        project = projects[0]
        proj_name = project[1:] if project.startswith("+") else project
        related = tasks_for_project(state, project)
        completed = [t for t in related if t.done]
        pending = [t for t in related if not t.done]

        def task_attr(t: Task) -> int:
            attr = curses.color_pair(1)
            if not t.done and t.date < state.today:
                attr = curses.color_pair(4)
            elif t.date == state.today:
                attr = curses.color_pair(2)
            if t.date > state.today and not t.done:
                attr |= curses.A_DIM
            if t.done:
                attr |= curses.A_DIM
            return attr

        prefix = "Part of project: "
        win.addnstr(row, 2, prefix, win_w - 4, curses.A_BOLD)
        if len(prefix) < win_w - 4:
            win.addnstr(row, 2 + len(prefix), proj_name[: win_w - 4 - len(prefix)], win_w - 4 - len(prefix))
        row += 1
        if row < win_h - 2:
            win.addnstr(row, 2, "", win_w - 4)
            row += 1
        sections = [("Pending:", pending), ("Complete:", completed)]
        for label, bucket in sections:
            if row >= win_h - 2:
                break
            win.addnstr(row, 4, label, win_w - 6, curses.A_BOLD)
            row += 1
            for t in bucket:
                if row >= win_h - 2:
                    break
                text_line = f"- {t.display_text()}"
                win.addnstr(row, 6, text_line[: win_w - 8], win_w - 8, task_attr(t))
                row += 1
            if row < win_h - 2:
                row += 1

    win.refresh()
    kstate = KeyState()
    try:
        read_key(win, kstate)
    except curses.error:
        pass


def show_welcome_screen(stdscr: curses.window, state: State) -> None:
    win_h, win_w, start_y, start_x = compact_modal_geometry(stdscr)
    win = curses.newwin(win_h, win_w, start_y, start_x)
    win.erase()
    win.border()
    title = " Welcome to tweeker "
    win.addnstr(0, max(1, (win_w - len(title)) // 2), title, len(title), curses.A_BOLD)

    messages = [
        "No tasks found in todo.txt.",
        "Welcome! tweeker is a week-first, keyboard-driven task list. Keep a small, focused set of tasks per day and keep them moving forward.",
        "",
        "Core keys: h/l move days, j/k move tasks, o/O add below/above, i edit, x toggle done, +/- adjust priority, Enter shows details, q quits.",
        "",
        "Suggested flow: start on today, press o to add a few tasks, mark them done with x as you progress, jump between days with h/l, and keep priorities tight with +/- to stay focused.",
        "Press any key to begin.",
    ]
    row = 2
    wrap_width = max(20, win_w - 6)
    for line in messages:
        if row >= win_h - 2:
            break
        if line:
            for segment in textwrap.wrap(line, width=wrap_width):
                if row >= win_h - 2:
                    break
                win.addnstr(row, 3, segment, win_w - 6)
                row += 1
        else:
            win.addnstr(row, 3, "", win_w - 6)
            row += 1

    win.refresh()
    kstate = KeyState()
    try:
        return read_key(win, kstate)
    except curses.error:
        return None


def show_config_errors(stdscr: curses.window, errors: list[str]) -> None:
    if not errors:
        return
    win_h, win_w, start_y, start_x = compact_modal_geometry(stdscr)
    win = curses.newwin(win_h, win_w, start_y, start_x)
    win.erase()
    win.border()
    title = " Config errors (defaults applied) "
    win.addnstr(0, max(1, (win_w - len(title)) // 2), title, len(title), curses.A_BOLD)
    row = 1
    wrap_width = max(20, win_w - 6)
    for err in errors:
        if row >= win_h - 2:
            break
        wrapped = textwrap.wrap(err, width=wrap_width)
        for seg in wrapped:
            if row >= win_h - 2:
                break
            win.addnstr(row, 3, f"- {seg}", win_w - 6)
            row += 1
    if row < win_h - 2:
        hint = "Press any key to continue with defaults."
        win.addnstr(win_h - 2, max(2, (win_w - len(hint)) // 2), hint, win_w - 4)
    win.refresh()
    kstate = KeyState()
    try:
        read_key(win, kstate)
    except curses.error:
        pass


def handle_command(stdscr: curses.window, state: State, config: Config) -> str:
    draw(stdscr, state, show_help=False)
    text = prompt_text(stdscr, ":", "")
    cmd = text.strip().lower()
    if not cmd:
        return ""
    if cmd in {"help", "h", "keys", "keybinds", ":help"}:
        show_keybinds(stdscr, config)
        return "help"
    return f"unknown command: {cmd}"


def insert_at(stdscr: curses.window, state: State, idx: int) -> None:
    draw(stdscr, state, show_help=False)
    text = prompt_text(stdscr, "new")
    if not text.strip():
        return
    done, priority, cleaned = parse_text_input(text)
    record_undo(state)
    items = state.tasks.setdefault(state.current_date(), [])
    task = Task(date=state.current_date(), text=cleaned, done=done, priority=priority)
    items.insert(idx, task)
    sort_tasks_for_date(state, state.current_date(), focus_task=task)


def insert_relative_to_tasks(stdscr: curses.window, state: State, targets: List[Task], *, after: bool) -> None:
    draw(stdscr, state, show_help=False)
    text = prompt_text(stdscr, "new")
    if not text.strip():
        return
    done, priority, cleaned = parse_text_input(text)
    record_undo(state)
    target_ids = {id(t) for t in targets}
    by_date: Dict[dt.date, list[Task]] = {}
    for date, tasks in state.tasks.items():
        for t in tasks:
            if id(t) in target_ids:
                by_date.setdefault(date, []).append(t)
    last_inserted = None
    for date, ref_tasks in by_date.items():
        items = state.tasks.setdefault(date, [])
        for ref in ref_tasks:
            try:
                pos = next(i for i, t in enumerate(items) if t is ref)
            except StopIteration:
                pos = len(items)
            insert_pos = pos + 1 if after else pos
            new_task = Task(date=date, text=cleaned, done=done, priority=priority)
            items.insert(insert_pos, new_task)
            last_inserted = new_task
        sort_tasks_for_date(state, date, focus_task=last_inserted)


def main(stdscr: curses.window) -> None:
    config, config_errors = load_config()
    curses.curs_set(0)
    curses.use_default_colors()
    curses.start_color()
    curses.init_pair(1, config.default_fg, -1)  # default text
    curses.init_pair(2, config.highlight_fg, -1)  # highlight text (today column + selection)
    curses.init_pair(3, config.default_fg, -1)  # status bar / misc
    curses.init_pair(4, curses.COLOR_RED, -1)  # overdue tasks
    stdscr.keypad(True)

    tasks = load_tasks(config.todo_file)
    today = dt.date.today()
    overdue_changed = handle_overdue_tasks(tasks, today, config.move_overdue_to_today)
    if overdue_changed:
        save_tasks(tasks, config.todo_file)
    week_start, today_idx, days = window_for_date(today, config)
    state = State(
        tasks=tasks,
        week_start=week_start,
        cursor_day=today_idx,
        today=today,
        day_order=day_names_for(week_start, days),
        first_weekday=config.first_weekday,
        config=config,
        last_save_time=time.monotonic(),
    )
    state.update_week_window(state.week_start, state.cursor_day)
    for d in list(state.tasks.keys()):
        sort_tasks_for_date(state, d)

    if config_errors:
        show_config_errors(stdscr, config_errors)

    show_welcome = not tasks
    welcome_shown = False
    pending_delete = False
    pending_copy = False
    status = ""
    kstate = KeyState()
    key_buffer: List[str] = []

    def is_action(key: object, action: str) -> bool:
        if key in config.keybinds.get(action, set()):
            return True
        if action == "details":
            fallback = {
                "\n",
                "\r",
                getattr(curses, "KEY_ENTER", None),
                10,
            }
            return key in fallback
        return False

    repeatable = (
        config.keybinds.get("delete", set())
        | config.keybinds.get("copy", set())
        | config.keybinds.get("goto_prefix", set())
        | config.keybinds.get("goto_date", set())
        | config.keybinds.get("jump_prefix", set())
        | config.keybinds.get("jump_date", set())
    )

    while True:
        # Block on too-small terminal until resized
        while True:
            h, w = stdscr.getmaxyx()
            min_w = max(MIN_WIDTH, MIN_COL_WIDTH * len(state.day_order))
            if w >= min_w and h >= MIN_HEIGHT:
                break
            stdscr.erase()
            msg = f"tweeker needs at least {min_w}x{MIN_HEIGHT}. current: {w}x{h}"
            hint = "resize your terminal to continue"
            stdscr.addnstr(h // 2 - 1, max(0, (w - len(msg)) // 2), msg[: max(0, w - 1)], max(0, w - 1), curses.A_BOLD)
            stdscr.addnstr(h // 2, max(0, (w - len(hint)) // 2), hint[: max(0, w - 1)], max(0, w - 1))
            stdscr.refresh()
            try:
                stdscr.getch()
            except curses.error:
                continue
        draw(stdscr, state, status)
        key = None
        if show_welcome and not welcome_shown:
            key = show_welcome_screen(stdscr, state)
            welcome_shown = True
        if key is None:
            try:
                key = read_key(stdscr, kstate, allow_repeat_keys=repeatable)
            except curses.error:
                continue
        key_buffer.append(format_key(key))
        status = ""
        hold_buffer = False

        if pending_delete:
            pending_delete = False
            if is_action(key, "delete"):
                yank_current(state)
                targets = [t for _, t in selected_entries(state)] or ([current_task(state)] if current_task(state) else [])
                record_undo(state)
                delete_tasks(state, targets)
                clear_selection(state)
                mark_dirty(state)
                status = "deleted"
            else:
                status = ""
            key_buffer.clear()
            continue

        if pending_copy:
            pending_copy = False
            if is_action(key, "copy"):
                yank_current(state)
                status = "copied"
            key_buffer.clear()
            continue

        if key == "\x1b":
            if state.selected_ids:
                clear_selection(state)
                status = "selection cleared"
            key_buffer.clear()
            continue

        if is_action(key, "quit"):
            break
        if is_action(key, "left"):
            if state.config.view_mode == "week":
                if state.cursor_day == 0:
                    state.update_week_window(state.week_start - dt.timedelta(days=7), cursor_day=len(state.day_order) - 1)
                else:
                    state.cursor_day = state.cursor_day - 1
            else:
                if state.cursor_day == 0:
                    state.update_week_window(state.week_start - dt.timedelta(days=1), cursor_day=0)
                else:
                    state.cursor_day = state.cursor_day - 1
            state.clamp_cursor()
            update_range_selection(state)
        elif is_action(key, "right"):
            if state.config.view_mode == "week":
                if state.cursor_day == len(state.day_order) - 1:
                    state.update_week_window(state.week_start + dt.timedelta(days=7), cursor_day=0)
                else:
                    state.cursor_day = state.cursor_day + 1
            else:
                if state.cursor_day == len(state.day_order) - 1:
                    state.update_week_window(state.week_start + dt.timedelta(days=1), cursor_day=len(state.day_order) - 1)
                else:
                    state.cursor_day = state.cursor_day + 1
            state.clamp_cursor()
            update_range_selection(state)
        elif is_action(key, "down"):
            items = state.current_tasks()
            if items:
                state.cursor_idx = min(state.cursor_idx + 1, len(items) - 1)
                update_range_selection(state)
        elif is_action(key, "up"):
            if state.cursor_idx > 0:
                state.cursor_idx -= 1
                update_range_selection(state)
        elif is_action(key, "edit"):
            targets = [t for _, t in selected_entries(state)]
            if targets:
                edit_tasks(stdscr, state, targets)
            else:
                insert_or_edit(stdscr, state)
            mark_dirty(state)
        elif is_action(key, "new_below"):
            targets = [t for _, t in selected_entries(state)]
            if targets:
                insert_relative_to_tasks(stdscr, state, targets, after=True)
            else:
                items = state.tasks.setdefault(state.current_date(), [])
                insert_at(stdscr, state, state.cursor_idx + 1 if items else 0)
            mark_dirty(state)
        elif is_action(key, "new_above"):
            targets = [t for _, t in selected_entries(state)]
            if targets:
                insert_relative_to_tasks(stdscr, state, targets, after=False)
            else:
                insert_at(stdscr, state, state.cursor_idx if state.current_tasks() else 0)
            mark_dirty(state)
        elif is_action(key, "toggle_done"):
            targets = [t for _, t in selected_entries(state)] or ([current_task(state)] if current_task(state) else [])
            if targets:
                record_undo(state)
            for task in targets:
                task.done = not task.done
                sort_tasks_for_date(state, task.date, focus_task=task)
            mark_dirty(state)
        elif is_action(key, "priority_up"):
            targets = [t for _, t in selected_entries(state)] or ([current_task(state)] if current_task(state) else [])
            if targets:
                record_undo(state)
            for task in targets:
                adjust_priority(task, -1)
                sort_tasks_for_date(state, task.date, focus_task=task)
            if targets:
                mark_dirty(state)
        elif is_action(key, "priority_down"):
            targets = [t for _, t in selected_entries(state)] or ([current_task(state)] if current_task(state) else [])
            if targets:
                record_undo(state)
            for task in targets:
                adjust_priority(task, 1)
                sort_tasks_for_date(state, task.date, focus_task=task)
            if targets:
                mark_dirty(state)
        elif is_action(key, "select_range"):
            cur = current_task(state)
            if not cur:
                status = "no task"
            else:
                if state.selection_mode == "range":
                    clear_selection(state)
                else:
                    state.selection_mode = "range"
                    state.selection_anchor = id(cur)
                    state.selected_ids = {id(cur)}
                    status = "range select"
        elif is_action(key, "select_single"):
            cur = current_task(state)
            if not cur:
                status = "no task"
            else:
                state.selection_mode = "multi"
                if id(cur) in state.selected_ids:
                    state.selected_ids.remove(id(cur))
                else:
                    state.selected_ids.add(id(cur))
                status = f"{len(state.selected_ids)} selected"
        elif is_action(key, "undo"):
            if state.undo_stack:
                state.redo_stack.append(snapshot_state(state))
                snap = state.undo_stack.pop()
                apply_snapshot(state, snap)
                mark_dirty(state)
                status = "undone"
            else:
                status = "nothing to undo"
            clear_selection(state)
        elif is_action(key, "redo"):
            if state.redo_stack:
                state.undo_stack.append(snapshot_state(state))
                snap = state.redo_stack.pop()
                apply_snapshot(state, snap)
                mark_dirty(state)
                status = "redone"
            else:
                status = "nothing to redo"
            clear_selection(state)
        elif is_action(key, "goto_date"):
            key_buffer.clear()
            draw(stdscr, state, show_help=False)
            text = prompt_text(stdscr, "go to date", "")
            if not text.strip():
                status = ""
            else:
                target = parse_jump(text, state.today)
                if target:
                    new_start, new_idx, _ = window_for_date(target, state.config)
                    state.update_week_window(new_start, cursor_day=new_idx)
                    state.cursor_idx = 0
                    status = target.isoformat()
                else:
                    status = "invalid date"
        elif is_action(key, "jump_date"):
            key_buffer.clear()
            draw(stdscr, state, show_help=False)
            text = prompt_text(stdscr, "move to date", "")
            if not text.strip():
                status = ""
            else:
                target = parse_jump(text, state.today)
                if target:
                    targets = [t for _, t in selected_entries(state)] or ([current_task(state)] if current_task(state) else [])
                    moved = move_tasks_to_date(state, target, targets)
                    status = f"moved {len(targets)}" if moved else "no task"
                else:
                    status = "invalid date"
        elif is_action(key, "delete"):
            pending_delete = True
            status = "pending dd"
            hold_buffer = True
        elif is_action(key, "copy"):
            pending_copy = True
            status = "pending yy"
            hold_buffer = True
        elif is_action(key, "paste_after"):
            pasted = paste_task(state, before=False)
            if pasted:
                mark_dirty(state)
                status = "pasted"
            else:
                status = "nothing to paste"
        elif is_action(key, "paste_before"):
            pasted = paste_task(state, before=True)
            if pasted:
                mark_dirty(state)
                status = "pasted"
            else:
                status = "nothing to paste"
        elif is_action(key, "details"):
            show_task_details(stdscr, state)
            status = ""
        elif is_action(key, "week_forward"):
            step = 7 if state.config.view_mode == "week" else effective_window_days(state.config)
            state.update_week_window(state.week_start + dt.timedelta(days=step))
            state.cursor_idx = 0
            state.clamp_cursor()
            update_range_selection(state)
        elif is_action(key, "week_back"):
            step = 7 if state.config.view_mode == "week" else effective_window_days(state.config)
            state.update_week_window(state.week_start - dt.timedelta(days=step))
            state.cursor_idx = 0
            state.clamp_cursor()
            update_range_selection(state)
        elif is_action(key, "goto_prefix"):
            day_hints = " ".join(f"{i+1}:{day[:3]}" for i, day in enumerate(state.day_order))
            hint_text = f"{' '.join(key_buffer)} {day_hints} g:Today".strip()
            draw(stdscr, state, hint_text, show_help=False)
            prompt_state = KeyState()
            try:
                first = read_key(stdscr, prompt_state, allow_repeat_keys=config.keybinds.get("goto_prefix", set()))
            except curses.error:
                first = ""
            key_buffer.append(format_key(first))
            draw(stdscr, state, f"{' '.join(key_buffer)} {day_hints} g:Today", show_help=False)
            if isinstance(first, str) and first.isdigit() and 1 <= int(first) <= len(state.day_order):
                target_idx = int(first) - 1
                state.cursor_day = target_idx % len(state.day_order)
                state.clamp_cursor()
                update_range_selection(state)
                status = state.day_order[state.cursor_day]
            elif first == "g":
                target = state.today
                new_start, new_idx, _ = window_for_date(target, state.config)
                state.update_week_window(new_start, cursor_day=new_idx)
                status = "today"
            elif first in ("\x1b",):
                status = ""
            else:
                status = ""
            key_buffer.clear()
        elif is_action(key, "jump_prefix"):
            day_hints = " ".join(f"{i+1}:{day[:3]}" for i, day in enumerate(state.day_order))
            hint_text = f"{' '.join(key_buffer)} {day_hints} t:Today".strip()
            draw(stdscr, state, hint_text, show_help=False)
            prompt_state = KeyState()
            try:
                first = read_key(stdscr, prompt_state, allow_repeat_keys=config.keybinds.get("jump_prefix", set()))
            except curses.error:
                first = ""
            key_buffer.append(format_key(first))
            draw(stdscr, state, f"{' '.join(key_buffer)} {day_hints} t:Today", show_help=False)
            if isinstance(first, str) and first.isdigit() and 1 <= int(first) <= len(state.day_order):
                target_idx = int(first) - 1
                target_date = state.week_start + dt.timedelta(days=target_idx)
            elif first == "t":
                target_date = state.today
            else:
                target_date = None
            if target_date:
                targets = [t for _, t in selected_entries(state)] or ([current_task(state)] if current_task(state) else [])
                moved = move_tasks_to_date(state, target_date, targets)
                status = target_date.isoformat() if moved else "no task"
            else:
                status = ""
            key_buffer.clear()
        elif is_action(key, "command"):
            status = handle_command(stdscr, state, config)
            key_buffer.clear()
        elif is_action(key, "search_forward"):
            draw(stdscr, state, show_help=False)
            q = prompt_text(stdscr, "search /")
            if q:
                state.search_query = q
                state.search_dir = 1
                found = find_match(state, q, 1)
                if found:
                    date, idx = found
                    new_start, new_idx, _ = window_for_date(date, state.config)
                    state.update_week_window(new_start, cursor_day=new_idx)
                    state.cursor_idx = idx
                    status = f"{q}"
                else:
                    status = "no match"
        elif is_action(key, "search_backward"):
            draw(stdscr, state, show_help=False)
            q = prompt_text(stdscr, "search ?")
            if q:
                state.search_query = q
                state.search_dir = -1
                found = find_match(state, q, -1)
                if found:
                    date, idx = found
                    new_start, new_idx, _ = window_for_date(date, state.config)
                    state.update_week_window(new_start, cursor_day=new_idx)
                    state.cursor_idx = idx
                    status = f"{q}"
                else:
                    status = "no match"
        elif is_action(key, "search_next"):
            if not state.search_query:
                status = "no search"
            else:
                found = find_match(state, state.search_query, 1)
                if found:
                    date, idx = found
                    new_start, new_idx, _ = window_for_date(date, state.config)
                    state.update_week_window(new_start, cursor_day=new_idx)
                    state.cursor_idx = idx
                    status = state.search_query
                else:
                    status = "no match"
        elif is_action(key, "search_prev"):
            if not state.search_query:
                status = "no search"
            else:
                found = find_match(state, state.search_query, -1)
                if found:
                    date, idx = found
                    new_start, new_idx, _ = window_for_date(date, state.config)
                    state.update_week_window(new_start, cursor_day=new_idx)
                    state.cursor_idx = idx
                    status = state.search_query
                else:
                    status = "no match"

        if not hold_buffer and status != "pending dd" and status != "pending yy":
            key_buffer.clear()
        if not status and key_buffer:
            status = " ".join(key_buffer)

        persist_tasks_if_needed(state)

    persist_tasks_if_needed(state, force=True)


if __name__ == "__main__":
    def wait_for_resize(stdscr: curses.window) -> None:
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            msg = "tweeker needs a larger terminal. resize to continue."
            hint = f"Tip: try at least {max(MIN_WIDTH, MIN_COL_WIDTH * len(DEFAULT_DAY_ORDER))}x{MIN_HEIGHT}."
            stdscr.addnstr(h // 2 - 1, max(0, (w - len(msg)) // 2), msg, max(0, w - 1), curses.A_BOLD)
            stdscr.addnstr(h // 2, max(0, (w - len(hint)) // 2), hint, max(0, w - 1))
            stdscr.refresh()
            stdscr.getch()
            h, w = stdscr.getmaxyx()
            if w >= max(MIN_WIDTH, MIN_COL_WIDTH * len(DEFAULT_DAY_ORDER)) and h >= MIN_HEIGHT:
                break

    while True:
        try:
            curses.wrapper(main)
            break
        except curses.error:
            curses.wrapper(wait_for_resize)
            continue
