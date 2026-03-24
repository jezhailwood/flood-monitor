"""Minimal REST API client.

This module provides `APIClient`, a helper for constructing URLs relative to a base API
endpoint and making HTTP GET requests using the `requests` library. Responses are
returned as parsed JSON.
"""

from typing import Any
from urllib.parse import urljoin

import requests


class APIClient:
    """HTTP client for simple GET requests against a REST API.

    The client constructs request URLs relative to a configured base URL, performs GET
    requests, and returns parsed JSON responses.

    This client does not currently handle authentication, retries, or custom headers.
    Network errors and non-2xx responses are raised by `requests`.

    Example:
        client = APIClient("https://api.example.com")
        data = client.get("users", 123, params={"profile": "full"})
    """

    def __init__(self, base_url: str, *, timeout: float = 10.0) -> None:
        """Initialise the client with a base API URL.

        The base URL is normalised to always end with exactly one trailing slash to
        ensure consistent URL joining.

        Args:
            base_url:
                Root URL of the API, eg "https://api.example.com".
            timeout:
                Default timeout in seconds applied to requests, unless overridden
                per call.
        """
        self.base_url = base_url.strip().rstrip("/") + "/"
        self.timeout = timeout
        self._session = requests.Session()

    def get(
        self,
        *path_segments: str | int,
        params: dict[str, str | int] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Make a GET request and return the parsed JSON response.

        The request URL is formed by joining `base_url` with `path_segments`.
        Each segment is converted to a string and stripped of leading and trailing
        slashes. If provided, `params` are included as query string parameters.

        Args:
            *path_segments:
                Path segments to append to the base URL,
                eg ("users", 123) -> ".../users/123".
            params:
                Optional query string parameters.
            timeout:
                Optional timeout override in seconds. If not provided, the client's
                default timeout is used.

        Returns:
            Parsed JSON from the response body.

        Raises:
            requests.HTTPError: If the response status code indicates an error.
            requests.RequestException: If a network-level error occurs.
            ValueError: If the response body is not valid JSON.
        """
        path = "/".join(str(p).strip("/") for p in path_segments)
        url = urljoin(self.base_url, path)
        effective_timeout = self.timeout if timeout is None else timeout
        response = self._session.get(url, params=params, timeout=effective_timeout)
        response.raise_for_status()
        return response.json()
