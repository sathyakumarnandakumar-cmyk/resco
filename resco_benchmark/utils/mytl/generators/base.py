import codecs
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional, TypeVar, Union
from xml.etree import ElementTree

from utils.mytl.typing import CodecsOpenParams

T = TypeVar("T", bound="Generator")


class Generator(metaclass=ABCMeta):
    """Base class for all SUMO configuration file generators.

    Parameters
    ----------
    path : str or pathlib.Path
        Path where the XML file will be saved.

    force : bool, optional (default=False)
        - If True, existing file under the `path` indicated above will
          be overwritten.
        - If False, unless the `path` points to an unallocated location,
          exception will be thrown.

    begin : int
       The beginning of the duration of the specified time period.

    end : int
        The end of the specified period of time.

    total_time : int
        Total duration of the generated file

    Attributes
    ----------
    root

    data : xml.etree.ElementTree.Element
        Root node in the XML tree representation from which the other
        sub-elements will branch.

    """

    OPEN_PARAMS_DEFAULT: Mapping[str, Any] = {"mode": "wb"}

    def __init__(
        self, path: Union[str, Path], *, force: bool = False, begin: int, end: int, total_time:int,  **kwargs
    ) -> None:
        self._path = None

        self.force = force
        self.path = path
        self.begin = begin
        self.end = end
        self.total_time = total_time

        self.data = ElementTree.Element(self.root)

    def __call__(
        self,
        open_params: CodecsOpenParams = None,
        **make_params,
    ) -> T:
        """Make an XML tree, then save it to a file.

        Parameters
        ----------
        open_params : dict_like, optional (default=None)
            Parameters passed to the `codecs.open` function.

        **make_params : dict
            Additional `make` parameters.

        """
        return self.make(**make_params).save(open_params=open_params)

    @property
    @abstractmethod
    def root(self) -> str:
        """Name of the root tag onto which other tags are created."""
        raise NotImplementedError

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if path.suffix != ".xml":
            raise ValueError(
                f"Output destination filename extension is `{path.suffix}`."
                "XML extension `.xml` is required."
            )
        if path.exists() and not self.force:
            raise FileExistsError(
                "An existing file is located at the indicated path. "
                "If you want to overwrite it anyway, set the parameter "
                "`force=True`."
            )

        self._path = path

    @abstractmethod
    def make(self, *args, **kwargs) -> T:
        """Write subelements to the XML tree.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self

    def save(self, open_params: Optional[CodecsOpenParams] = None) -> T:
        """Save XML element-tree to a file.

        Parameters
        ----------
        open_params : dict_like, optional (default=None)
            Parameters passed to the `codecs.open` function.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        if open_params is None:
            open_params = self.OPEN_PARAMS_DEFAULT
        else:
            open_params = {
                **self.OPEN_PARAMS_DEFAULT,
                **open_params,
            }

        data = ElementTree.tostring(self.data)
        with codecs.open(str(self.path), **open_params) as file:
            file.write(data)

        return self
