# tweeker

tweeker is a week-first, keyboard-driven terminal task list built around a lightweight `todo.txt`-style file. It renders a seven-day grid, keeps your task set small, and focuses on moving items forward day by day.

## Quick start

```bash
python tweeker.py
```

- Requires Python 3.10+ and a curses-capable terminal.
- By default tasks are read from `todo.txt` in the repo; you can point to another file via config.

## Task format

Each line is a task for a specific date:

- `YYYY-MM-DD Task text` — an active task.
- `x YYYY-MM-DD Task text` — a completed task.
- `(A) YYYY-MM-DD Task text` — priority A (A–Z supported).

Blank lines are ignored.

## Configuration

Config lives at `~/.config/tweeker/config.ini`. Invalid entries fall back to defaults and are reported in a small popup on startup.

```ini
[general]
show_statusbar = true
move_overdue_to_today = false
week_starts_on_sunday = false
todo_file = ~/todo/tweeker.txt
archive_file = ~/todo/archive.txt
archive_completed_after_days = 0
save_debounce_seconds = 0.3
expand_overflow_on_hover = true
days_on_screen = 5
view_mode = scroll

[colors]
default_fg = white
highlight_fg = cyan

[keybinds]
quit = q
left = h, KEY_LEFT
right = l, KEY_RIGHT
down = j, KEY_DOWN
up = k, KEY_UP
new_below = o
new_above = O
edit = i
toggle_done = x
priority_up = +
priority_down = -
delete = d
copy = y
paste_after = p
paste_before = P
week_forward = w
week_back = b
goto_prefix = g
goto_date = G
jump_prefix = t
jump_date = T
search_forward = /
search_backward = ?
search_next = n
search_prev = N
command = :
details = KEY_ENTER, ENTER
```

Notes:
- `todo_file` can be any path (directories are rejected). Missing files are created on save.
- `week_starts_on_sunday` moves the calendar to a Sunday-first layout; default is Monday-first.
- `archive_completed_after_days` moves completed tasks older than N days into `archive_file` (0 disables).
- `archive_file` defaults to `archive.txt` alongside your `todo_file` if not set.
- `save_debounce_seconds` batches disk writes; tasks always flush on exit or after the debounce window.
- `expand_overflow_on_hover` wraps long task text over multiple lines when the cursor is on it.
- `days_on_screen` controls how many consecutive days are shown at once (default 5); horizontal navigation scrolls the window day-by-day.
- `view_mode` chooses `scroll` (shows yesterday/today/forward with `days_on_screen` and scrolls a day at the edges) or `week` (always a 7-day week; edges jump a full week).
- Colors accept `black, red, green, yellow, blue, magenta, cyan, white, default` or an integer color ID if your terminal supports it.

## Core keys (defaults)

- Movement: `h`/`l` (days), `j`/`k` (tasks)
- Add/edit: `o`/`O` (new below/above), `i` (edit)
- Status: `x` (toggle done), `+`/`-` (priority up/down)
- Navigation: `w`/`b` (week forward/back), `g` (go to day by prefix), `t` (jump to day), `/`/`?` search, `n`/`N` next/prev match
- Clipboard: `y` (copy), `p`/`P` (paste after/before), `d` (delete)
- Info: `Enter` shows details, `:` opens command prompt, `q` quits

Press `?` in the app to see the keybinds window with your configured bindings.
