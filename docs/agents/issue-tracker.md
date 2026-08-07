# Issue tracker: Local Markdown

Issues and research maps for this repository live as versioned Markdown files
under `.scratch/`. This keeps the complete research plan in Git and avoids
requiring an external issue-tracker write.

## Conventions

- One effort per directory: `.scratch/<effort-slug>/`.
- A wayfinding map is `.scratch/<effort-slug>/map.md`.
- Child tickets are `.scratch/<effort-slug>/issues/<NN>-<slug>.md`.
- A ticket records `Type: research|prototype|grilling|task`, `Status:
  open|claimed|resolved`, and `Blocked by:` near the top.
- Resolution evidence is appended under `## Answer`; the map receives only a
  one-line context pointer under `## Decisions so far`.
- Open, unblocked, unclaimed tickets form the frontier; lowest number wins.

## Wayfinding operations

- **Claim:** change `Status: open` to `Status: claimed` and commit before work.
- **Resolve:** append the answer and evidence links, change status to
  `resolved`, append one gist/link to the map, and commit.
- **Blocking:** a ticket is unblocked only when every ticket named by
  `Blocked by:` is `resolved`.
- **Fog:** questions not yet sharp enough to ticket stay only in the map's
  `Not yet specified` section.

GitHub remains the source remote, but this tracker does not create or mutate
GitHub Issues unless the user separately authorizes switching trackers.
