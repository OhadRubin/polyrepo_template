"""
- description:
    Expose launch-layer helpers that turn experiment plans into rendered scripts
    and submit those scripts through queue-shaped backends.
- usage:
    uv run python -c "import polyrepo_launch; print(polyrepo_launch.__all__)"
    # Import submodules such as polyrepo_launch.generator from launcher code.
- user_story:
    content:
        As a launcher author, Ohad wants planning, execution shaping, and script
        generation helpers to live outside the core polyrepo sync package so the
        sync contract stays small while launchers can share queue-oriented code.
    was_generated_via_skill: false
"""

__all__ = []
