# History was rewritten before the baseline branch

Three earlier commits contained a personal home-directory path in a notebook, and all nine commits had personal author metadata. We rewrote the whole history with `git-filter-repo` so every commit uses one repository identity (see ADR 0003) and the path became `../..`. We chose this over allowlisting the old findings or scanning only new commits, so the full-history privacy scan can run with no exceptions. The cost is new commit IDs on every branch, a force push to the remote, and old clones that still hold the original history.
