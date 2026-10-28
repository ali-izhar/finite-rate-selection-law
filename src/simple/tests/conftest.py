"""Keep Hypothesis's storage out of the repository."""
import tempfile

from hypothesis.configuration import set_hypothesis_home_dir

set_hypothesis_home_dir(tempfile.mkdtemp(prefix="hypothesis-"))
