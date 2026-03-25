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
Type "export PATH=/Users/devwhodevs/Documents/engraph/target/release:$PATH"
Enter
Sleep 500ms
Show

# Title
Type "# engraph — local knowledge graph for AI agents"
Enter
Sleep 1s

# Index
Type "engraph index /tmp/engraph-demo-vault"
Enter
Sleep 2s

Type ""
Enter
Sleep 500ms

# Search
Type "# Hybrid search: semantic + keyword + graph expansion"
Enter
Sleep 800ms

Type "engraph search 'how does authentication work' --explain -n 5"
Enter
Sleep 4s

Type ""
Enter
Sleep 500ms

# Person context
Type "# Person context bundle — note + mentions + connections"
Enter
Sleep 800ms

Type "engraph context who 'Sarah Chen'"
Enter
Sleep 3s

Type ""
Enter
Sleep 500ms

# Status
Type "engraph status"
Enter
Sleep 2s

Sleep 3s
