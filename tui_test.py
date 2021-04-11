import curses

w = curses.initscr()
i = 0
row = 1
col = 1
while (i < 100):
    w.addstr(1, 1, str(i))
    w.refresh()
    row += 1
    col += 1
    i += 1
    curses.napms(500)

curses.endwin()
