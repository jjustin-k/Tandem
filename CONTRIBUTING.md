# Contributing to Tandem

Thanks for helping improve Tandem. Contributions can include source code, tests, documentation, bug reports, and ideas for the project.

## Start with an issue

Every contribution branch must have a corresponding GitHub issue, and the branch must be linked to that issue. This gives each change a clear purpose and keeps its discussion and review together.

1. Search the [existing issues](https://github.com/jjustin-k/Tandem/issues) to see whether the work is already being discussed.
2. If no issue covers the work, [open an issue](https://github.com/jjustin-k/Tandem/issues/new) describing the problem or proposed change. For a substantial change, discuss the approach before starting implementation.
3. Create the branch from the issue page using GitHub's **Create a branch** action, or use **Link a branch** to associate an existing branch with the issue. This is the preferred way to ensure the branch is linked. If creating it locally, use a descriptive name that includes the issue number, such as `27-update-contributing-guide`, then link it from the issue page.
4. Keep the work for that issue on its linked branch. If the scope changes substantially, update the issue or open a separate one and branch.

Do not start contribution work on an unlinked branch. If you have already created one, link it to the issue before opening a pull request.

## Make your change

- Keep changes focused on the issue and explain any relevant design or documentation decisions in the issue or pull request.
- Follow the conventions already present in the files you edit. Preserve the existing structure and formatting of project documents.
- Update relevant documentation when behavior, requirements, or project instructions change.
- Add or update tests when appropriate for the change. Do not claim a check passed unless you ran it.
- For LaTeX or Markdown documents that build to PDF, use the repository's `make` targets where applicable and avoid committing generated build files unless the issue calls for them.
- Review your diff before submitting. Remove unrelated edits, generated artifacts, credentials, and local configuration.

## Open a pull request

1. Push your issue-linked branch to GitHub.
2. Open a pull request against the repository's default branch. Make sure the pull request is associated with the same issue; include `Closes #<issue-number>` in its description when the change will close that issue.
3. Describe what changed, why it changed, and how you checked it. Mention any checks you could not run and any remaining questions.
4. Respond to review feedback and keep the issue and pull request up to date as the work evolves.

Pull requests may be revised before they are merged. A maintainer may ask for changes, clarification, or a smaller scope to keep the project coherent.

## Report bugs and suggest improvements

Open an issue with a clear title and enough context for someone else to understand and reproduce the concern. For a bug, include the steps to reproduce it, what you expected to happen, what happened instead, and relevant environment or document details. For a proposal, explain the problem it addresses and the outcome you have in mind. Avoid including passwords, tokens, personal information, or other sensitive data.

## Be respectful

Keep project discussions constructive, considerate, and focused on the work. Assume good intent, welcome differing perspectives, and give feedback about ideas and changes rather than people. Report concerns about harmful or inappropriate behavior to a project maintainer.

## Questions

If you are unsure where to begin or whether a change fits the project, raise the question in a relevant GitHub issue before doing substantial work.
