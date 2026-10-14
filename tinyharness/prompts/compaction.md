Summarize the conversation in the <conversation> tags above as an execution checkpoint. The agent will
continue from this checkpoint alone, with only the most recent messages still in view. Use exactly these
headings:

## Objective
What the user asked for, including exact constraints and preferences.

## Hard Constraints
Instructions, safety limits, compatibility requirements, and things the user explicitly rejected.

## Execution State
### Completed — Do Not Repeat
Work already finished, including completed Agent or Team calls and the useful results they returned.
### Active
Work started but not finished, with its exact current state.
### Pending
Work that has not started and is still required.

## Findings and Decisions
Facts learned (exact numbers, names, paths, commands, and errors) and decisions with their reasons. Results
not recorded here are lost.

## Artifacts and Files
Each relevant path and whether it was read, changed, created, or only proposed.

## Verification
Checks already run and their outcomes; clearly distinguish unverified work.

## Next Action
Exactly one concrete next action. If the task is complete, write "Report completion."

Reconcile state rather than copying the original request into Pending. Anything proved finished belongs in
Completed — Do Not Repeat and must not also appear in Active, Pending, or Next Action. A completed Agent or
Team call is prior work to reuse, never a reason to create the team again. Keep every heading even when empty
(write "(none)"). Be concise but lose no facts. No preamble, and do not mention that this is a summary.
