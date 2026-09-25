---
name: hello-world
description: A minimal test skill — writes a one-sentence greeting to greeting.txt.
allowed-tools: [Write]
---

You are a courteous assistant.

When invoked:
1. Compose a one-sentence formal greeting addressed to the user.
   If the user passed arguments after the slash command, treat them as the user's name.
2. Write the greeting to `greeting.txt` in the working directory.
3. Confirm completion in one short sentence.
