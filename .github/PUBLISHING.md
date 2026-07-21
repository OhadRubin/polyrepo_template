# Publishing

The package name is `polyrepo-template`.

To claim the PyPI project name, create a pending trusted publisher on PyPI with
these fields:

```text
owner: OhadRubin
repository: polyrepo_template
workflow: publish.yml
environment: pypi
```

Publish a GitHub Release for a tag such as `v0.1.0`. The release workflow builds
the distributions with `uv`, runs the unit tests, checks the package metadata,
and publishes to PyPI through trusted publishing.
