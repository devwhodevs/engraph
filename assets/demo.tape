Output assets/demo.gif
Set Shell zsh
Set FontSize 15
Set Width 1100
Set Height 700
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 50ms
Set WindowBar Colorful
Set WindowBarSize 40

Hide
Type "export PATH=/opt/homebrew/bin:$PATH"
Enter
Sleep 500ms
Show

# Scene 1: Index
Type "# Index an Obsidian vault (Metal GPU, ~70s for 88 files)"
Sleep 1.5s
Ctrl+U
Sleep 200ms

Type "engraph index /tmp/engraph-demo-vault"
Enter
Sleep 3s

Type ""
Enter
Sleep 300ms

# Scene 2: 4-lane search with --explain
Type "# 4-lane hybrid search: semantic + keyword + graph + reranker"
Sleep 1.5s
Ctrl+U
Sleep 200ms

Type "engraph search 'how does authentication work' --explain"
Enter
Sleep 5s

Type ""
Enter
Sleep 300ms

# Scene 3: Person context
Type "# Rich context bundles for AI agents"
Sleep 1.5s
Ctrl+U
Sleep 200ms

Type "engraph context who 'Sarah Chen'"
Enter
Sleep 4s

Type ""
Enter
Sleep 300ms

# Scene 4: Status
Type "# Index stats + intelligence state"
Sleep 1.5s
Ctrl+U
Sleep 200ms

Type "engraph status"
Enter
Sleep 3s

Sleep 3s
