' Launch the brain watcher without a console window.
' cscript/wscript is the only reliable way to keep a .cmd hidden at logon;
' a Startup shortcut set to "Minimized" still flashes a window each boot.
Set shell = CreateObject("WScript.Shell")
shell.Run """D:\Projects\W1z4rDV1510n\scripts\aws\run_brain_watch.cmd""", 0, False
